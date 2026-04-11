import * as vscode from 'vscode';
import { GatewayClient } from './client';
import { GatewayConfig, OpenAIChatCompletionRequest } from './types';
import { ThinkingParser } from './thinking';

type OpenAIRequestMessage = Record<string, unknown>;
type UserContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

interface ThinkingPiece {
  t: string;
  c: string;
}

interface StreamStats {
  totalContent: string;
  totalToolCalls: number;
  totalTextParts: number;
  hadThinking: boolean;
  thinkingForceClosed: boolean;
}

/**
 * Language model provider for OpenAI-compatible inference servers
 */
export class GatewayProvider implements vscode.LanguageModelChatProvider {
  // Token accounting defaults / floors. Config values override these when present.
  private static readonly DEFAULT_CONTEXT_TOKENS = 32768;
  private static readonly DEFAULT_OUTPUT_TOKENS = 2048;
  private static readonly FALLBACK_OUTPUT_TOKENS = 4096;
  private static readonly MIN_OUTPUT_TOKENS = 64;
  private static readonly CONTEXT_BUFFER_TOKENS = 256;
  private static readonly ADJUST_TOKEN_BUFFER = 256;
  private static readonly INPUT_OVERHEAD_RATIO = 1.2;
  private static readonly CHARS_PER_TOKEN = 4;
  private static readonly TOOL_RESPONSE_DELAY_MS = 3000;
  private static readonly DEFAULT_REQUEST_TIMEOUT_MS = 60000;
  private static readonly DEFAULT_TEMPERATURE = 0.7;

  private readonly client: GatewayClient;
  private config: GatewayConfig;
  private readonly outputChannel: vscode.OutputChannel;
  // Store tool schemas for the current request to fill missing required properties
  private readonly currentToolSchemas: Map<string, unknown> = new Map();
  // Track if we've shown the welcome notification this session
  private hasShownWelcomeNotification = false;

  constructor(context: vscode.ExtensionContext) {
    this.outputChannel = vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
    this.config = this.loadConfig();
    this.client = new GatewayClient(this.config, (msg) => this.outputChannel.appendLine(msg));

    context.subscriptions.push(
      vscode.workspace.onDidChangeConfiguration((e: vscode.ConfigurationChangeEvent) => {
        if (e.affectsConfiguration('github.copilot.llm-gateway')) {
          this.outputChannel.appendLine('Configuration changed, reloading...');
          this.reloadConfig();
        }
      })
    );
  }

  /**
   * Map VS Code message role to OpenAI role string
   */
  private mapRole(role: vscode.LanguageModelChatMessageRole): string {
    if (role === vscode.LanguageModelChatMessageRole.Assistant) {
      return 'assistant';
    }
    return 'user';
  }

  /**
   * Convert a tool result part to OpenAI format
   */
  private convertToolResultPart(part: vscode.LanguageModelToolResultPart): OpenAIRequestMessage {
    return {
      tool_call_id: part.callId,
      role: 'tool',
      content: typeof part.content === 'string' ? part.content : JSON.stringify(part.content),
    };
  }

  /**
   * Convert a tool call part to OpenAI format
   */
  private convertToolCallPart(part: vscode.LanguageModelToolCallPart): OpenAIRequestMessage {
    return {
      id: part.callId,
      type: 'function',
      function: {
        name: part.name,
        arguments: JSON.stringify(part.input),
      },
    };
  }

  /**
   * Encode a VS Code data part as a base64 data: URL suitable for OpenAI image_url input.
   */
  private encodeImageDataPart(part: vscode.LanguageModelDataPart): string {
    const base64Data = btoa(String.fromCodePoint(...part.data));
    return `data:${part.mimeType};base64,${base64Data}`;
  }

  /**
   * Get default value for a JSON schema type
   */
  private getDefaultForType(schema: Record<string, unknown> | null | undefined): unknown {
    if (!schema?.type) {
      return null;
    }

    switch (schema.type) {
      case 'string':
        return schema.default ?? '';
      case 'number':
      case 'integer':
        return schema.default ?? 0;
      case 'boolean':
        return schema.default ?? false;
      case 'array':
        return schema.default ?? [];
      case 'object':
        return schema.default ?? {};
      case 'null':
        return null;
      default:
        if (Array.isArray(schema.type)) {
          if (schema.type.includes('null')) {
            return null;
          }
          for (const t of schema.type) {
            if (t !== 'null') {
              return this.getDefaultForType({ ...schema, type: t });
            }
          }
        }
        return null;
    }
  }

  /**
   * Fill in missing required properties with default values based on the tool schema
   */
  private fillMissingRequiredProperties(
    args: Record<string, unknown>,
    toolSchema: Record<string, unknown> | null | undefined
  ): Record<string, unknown> {
    if (!toolSchema?.required || !Array.isArray(toolSchema.required)) {
      return args;
    }

    const properties = (toolSchema.properties || {}) as Record<string, Record<string, unknown>>;
    const filledArgs = { ...args };
    const filledProperties: string[] = [];

    for (const requiredProp of toolSchema.required as string[]) {
      if (!(requiredProp in filledArgs)) {
        const propSchema = properties[requiredProp];
        const defaultValue = this.getDefaultForType(propSchema);
        filledArgs[requiredProp] = defaultValue;
        filledProperties.push(`${requiredProp}=${JSON.stringify(defaultValue)}`);
      }
    }

    if (filledProperties.length > 0) {
      this.outputChannel.appendLine(`  AUTO-FILLED missing required properties: ${filledProperties.join(', ')}`);
    }

    return filledArgs;
  }

  /**
   * Estimate token count for a message (rough: CHARS_PER_TOKEN per token).
   */
  private estimateMessageTokens(message: OpenAIRequestMessage): number {
    let text = '';
    if (typeof message.content === 'string') {
      text = message.content;
    } else if (message.content) {
      text = JSON.stringify(message.content);
    }
    if (message.tool_calls) {
      text += JSON.stringify(message.tool_calls);
    }
    return Math.ceil(text.length / GatewayProvider.CHARS_PER_TOKEN);
  }

  /**
   * Truncate messages to fit within a token limit.
   * Strategy: Keep the first message (usually system prompt) and the most recent messages.
   */
  private truncateMessagesToFit(
    messages: OpenAIRequestMessage[],
    maxTokens: number
  ): OpenAIRequestMessage[] {
    if (messages.length === 0) {
      return messages;
    }

    let totalTokens = 0;
    const messageTokens: number[] = [];
    for (const msg of messages) {
      const tokens = this.estimateMessageTokens(msg);
      messageTokens.push(tokens);
      totalTokens += tokens;
    }

    if (totalTokens <= maxTokens) {
      return messages;
    }

    this.outputChannel.appendLine(`Context overflow: ${totalTokens} tokens > ${maxTokens} limit. Truncating...`);

    const result: OpenAIRequestMessage[] = [messages[0]];
    let usedTokens = messageTokens[0];

    const recentMessages: OpenAIRequestMessage[] = [];
    for (let i = messages.length - 1; i > 0; i--) {
      const msgTokens = messageTokens[i];
      if (usedTokens + msgTokens <= maxTokens) {
        recentMessages.unshift(messages[i]);
        usedTokens += msgTokens;
      } else {
        break;
      }
    }

    result.push(...recentMessages);

    this.outputChannel.appendLine(`Truncated: kept ${result.length}/${messages.length} messages, ~${usedTokens} tokens`);

    return result;
  }

  /**
   * Count occurrences of a literal character in a string.
   */
  private countChar(str: string, char: string): number {
    const escapePattern = /[.*+?^${}()|[\]\\]/g;
    const escapedChar = char.replaceAll(escapePattern, String.raw`\$&`);
    const regex = new RegExp(escapedChar, 'g');
    return (str.match(regex) ?? []).length;
  }

  /**
   * Balance unclosed braces/brackets in a JSON string
   */
  private balanceBrackets(str: string): string {
    let result = str;
    const missingBrackets = this.countChar(result, '[') - this.countChar(result, ']');
    const missingBraces = this.countChar(result, '{') - this.countChar(result, '}');

    result += ']'.repeat(Math.max(0, missingBrackets));
    result += '}'.repeat(Math.max(0, missingBraces));

    return result;
  }

  /**
   * Attempt to repair truncated or malformed JSON arguments
   */
  private tryRepairJson(jsonStr: string): unknown {
    if (!jsonStr || jsonStr.trim() === '') {
      return {};
    }

    try {
      return JSON.parse(jsonStr);
    } catch {
      // Fall through to repair attempts.
    }

    let repaired = jsonStr.trim();
    repaired = this.balanceBrackets(repaired);
    repaired = repaired.replaceAll(/,\s*([}\]])/g, '$1');

    if (this.countChar(repaired, '"') % 2 !== 0) {
      repaired += '"';
      repaired = this.balanceBrackets(repaired);
    }

    try {
      return JSON.parse(repaired);
    } catch {
      this.outputChannel.appendLine(`JSON repair failed. Original: ${jsonStr}`);
      this.outputChannel.appendLine(`Repaired attempt: ${repaired}`);
      return null;
    }
  }

  /**
   * Provide language model information - fetches available models from inference server
   */
  async provideLanguageModelChatInformation(
    options: { silent: boolean },
    _token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelChatInformation[]> {
    try {
      this.outputChannel.appendLine('Fetching models from inference server...');
      const response = await this.client.fetchModels();

      const models = response.data.map((model) => ({
        id: model.id,
        name: model.id,
        family: 'llm-gateway',
        maxInputTokens: this.config.defaultMaxTokens,
        maxOutputTokens: this.config.defaultMaxOutputTokens,
        version: '1.0.0',
        capabilities: {
          imageInput: this.config.enableImageInput,
          toolCalling: this.config.enableToolCalling,
        },
      } as vscode.LanguageModelChatInformation));

      this.outputChannel.appendLine(`Found ${models.length} models: ${models.map((m) => m.id).join(', ')}`);
      return models;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.outputChannel.appendLine(`ERROR: Failed to fetch models: ${errorMessage}`);

      if (!options.silent) {
        this.promptOpenSettings(
          `GitHub Copilot LLM Gateway: Failed to fetch models. ${errorMessage}`
        );
      }

      return [];
    }
  }

  private promptOpenSettings(message: string): void {
    vscode.window
      .showErrorMessage(message, 'Open Settings')
      .then(
        (selection: string | undefined) => {
          if (selection === 'Open Settings') {
            void vscode.commands.executeCommand(
              'workbench.action.openSettings',
              'github.copilot.llm-gateway'
            );
          }
        },
        (err: unknown) => {
          this.outputChannel.appendLine(
            `Failed to show settings prompt: ${err instanceof Error ? err.message : String(err)}`
          );
        }
      );
  }

  /**
   * Process a message part using duck-typing for older VS Code versions
   */
  private processPartDuckTyped(
    part: unknown,
    toolResults: OpenAIRequestMessage[],
    toolCalls: OpenAIRequestMessage[]
  ): void {
    const anyPart = part as Record<string, unknown>;
    if ('callId' in anyPart && 'content' in anyPart && !('name' in anyPart)) {
      this.outputChannel.appendLine(`  Found tool result (duck-typed): callId=${anyPart.callId}`);
      toolResults.push({
        tool_call_id: anyPart.callId,
        role: 'tool',
        content: typeof anyPart.content === 'string' ? anyPart.content : JSON.stringify(anyPart.content),
      });
    } else if ('callId' in anyPart && 'name' in anyPart && 'input' in anyPart) {
      this.outputChannel.appendLine(`  Found tool call (duck-typed): callId=${anyPart.callId}, name=${anyPart.name}`);
      toolCalls.push({
        id: anyPart.callId,
        type: 'function',
        function: { name: anyPart.name, arguments: JSON.stringify(anyPart.input) },
      });
    }
  }

  /**
   * Convert a single VS Code message to its OpenAI wire format representation.
   */
  private convertMessage(msg: vscode.LanguageModelChatMessage): OpenAIRequestMessage[] {
    const role = this.mapRole(msg.role);
    const toolResults: OpenAIRequestMessage[] = [];
    const toolCalls: OpenAIRequestMessage[] = [];
    const userContent: UserContentPart[] = [];
    let textContent = '';

    for (const part of msg.content) {
      if (part instanceof vscode.LanguageModelTextPart) {
        userContent.push({ type: 'text', text: part.value });
        textContent += part.value;
      } else if (part instanceof vscode.LanguageModelToolResultPart) {
        this.outputChannel.appendLine(`  Found tool result: callId=${part.callId}`);
        toolResults.push(this.convertToolResultPart(part));
      } else if (part instanceof vscode.LanguageModelToolCallPart) {
        this.outputChannel.appendLine(`  Found tool call: callId=${part.callId}, name=${part.name}`);
        toolCalls.push(this.convertToolCallPart(part));
      } else if (part instanceof vscode.LanguageModelDataPart) {
        if (this.config.enableImageInput) {
          if (part.mimeType.startsWith('image/')) {
            const url = this.encodeImageDataPart(part);
            userContent.push({ type: 'image_url', image_url: { url } });
            this.outputChannel.appendLine(
              `  Added image data part as base64 URL: mimeType=${part.mimeType}, size=${part.data.length} bytes, urlLength=${url.length}`
            );
          }
        } else {
          this.outputChannel.appendLine(
            `  Skipping data part: mimeType=${part.mimeType}, size=${part.data.length} bytes. (Please enable github.copilot.llm-gateway.enableImageInput in settings)`
          );
        }
      } else {
        this.processPartDuckTyped(part, toolResults, toolCalls);
      }
    }

    const result: OpenAIRequestMessage[] = [];
    if (toolCalls.length > 0) {
      result.push({ role: 'assistant', content: textContent || null, tool_calls: toolCalls });
    } else if (toolResults.length > 0) {
      result.push(...toolResults);
    } else if (userContent.length > 0) {
      result.push({ role, content: userContent });
    } else if (textContent) {
      result.push({ role, content: textContent });
    }
    return result;
  }

  /**
   * Calculate safe max output tokens based on input estimate
   */
  private calculateSafeMaxOutputTokens(estimatedInputTokens: number, toolsOverhead: number): number {
    const modelMaxContext = this.config.defaultMaxTokens || GatewayProvider.DEFAULT_CONTEXT_TOKENS;
    const totalEstimatedTokens = estimatedInputTokens + toolsOverhead;
    const conservativeInputEstimate = Math.ceil(totalEstimatedTokens * GatewayProvider.INPUT_OVERHEAD_RATIO);

    const safeMaxOutputTokens = Math.min(
      this.config.defaultMaxOutputTokens || GatewayProvider.DEFAULT_OUTPUT_TOKENS,
      Math.floor(modelMaxContext - conservativeInputEstimate - GatewayProvider.CONTEXT_BUFFER_TOKENS)
    );

    return Math.max(GatewayProvider.MIN_OUTPUT_TOKENS, safeMaxOutputTokens);
  }

  /**
   * Build tools configuration for request
   */
  private buildToolsConfig(
    options: vscode.ProvideLanguageModelChatResponseOptions
  ): OpenAIRequestMessage[] | undefined {
    if (!this.config.enableToolCalling || !options.tools || options.tools.length === 0) {
      return undefined;
    }

    this.currentToolSchemas.clear();

    return options.tools.map((tool) => {
      this.outputChannel.appendLine(`Tool: ${tool.name}`);
      this.outputChannel.appendLine(`  Description: ${tool.description?.substring(0, 100) || 'none'}...`);

      const schema = tool.inputSchema as Record<string, unknown> | undefined;
      this.currentToolSchemas.set(tool.name, schema);

      if (schema?.required && Array.isArray(schema.required)) {
        this.outputChannel.appendLine(`  Required properties: ${(schema.required as string[]).join(', ')}`);
      }

      return {
        type: 'function',
        function: { name: tool.name, description: tool.description, parameters: tool.inputSchema },
      };
    });
  }

  /**
   * Process a single tool call from the stream
   */
  private processToolCall(
    toolCall: { id: string; name: string; arguments: string },
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): void {
    this.outputChannel.appendLine(`\n=== TOOL CALL RECEIVED ===`);
    this.outputChannel.appendLine(`  ID: ${toolCall.id}`);
    this.outputChannel.appendLine(`  Name: ${toolCall.name}`);
    this.outputChannel.appendLine(
      `  Raw arguments: ${toolCall.arguments.substring(0, 1000)}${toolCall.arguments.length > 1000 ? '...' : ''}`
    );

    let args = this.tryRepairJson(toolCall.arguments) as Record<string, unknown> | null;

    if (args === null) {
      this.outputChannel.appendLine(`  ERROR: Failed to parse tool call arguments`);
      this.outputChannel.appendLine(`  Full arguments: ${toolCall.arguments}`);
      args = {};
    } else {
      const argKeys = Object.keys(args);
      this.outputChannel.appendLine(`  Parsed argument keys: ${argKeys.length > 0 ? argKeys.join(', ') : '(none)'}`);
    }

    const toolSchema = this.currentToolSchemas.get(toolCall.name) as Record<string, unknown> | undefined;
    if (toolSchema) {
      args = this.fillMissingRequiredProperties(args, toolSchema);
    }

    this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
    progress.report(new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.name, args));
  }

  /**
   * Handle empty response from model
   */
  private async handleEmptyResponse(
    model: vscode.LanguageModelChatInformation,
    inputText: string,
    messageCount: number,
    toolCount: number,
    token: vscode.CancellationToken,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): Promise<void> {
    const inputTokenCount = await this.provideTokenCount(model, inputText, token);
    const modelMaxContext = this.config.defaultMaxTokens || GatewayProvider.DEFAULT_CONTEXT_TOKENS;

    this.outputChannel.appendLine(`WARNING: Model returned empty response with no tool calls.`);
    this.outputChannel.appendLine(`  Input tokens estimated: ${inputTokenCount}`);
    this.outputChannel.appendLine(`  Messages in conversation: ${messageCount}`);
    this.outputChannel.appendLine(`  Tools provided: ${toolCount}`);

    const errorHint = toolCount > 0
      ? `The model returned an empty response. This typically indicates the model failed to generate valid output with tool calling enabled. Check the inference server logs for errors.`
      : `The model returned an empty response. Check the inference server logs for details.`;

    this.outputChannel.appendLine(`  Issue: ${errorHint}`);

    const errorMessage = `I was unable to generate a response. ${errorHint}\n\n` +
      `Diagnostic info:\n- Model: ${model.id}\n- Tools provided: ${toolCount}\n` +
      `- Estimated input tokens: ${inputTokenCount}\n- Context limit: ${modelMaxContext}\n\n` +
      `Check the "GitHub Copilot LLM Gateway" output panel for detailed logs.`;

    progress.report(new vscode.LanguageModelTextPart(errorMessage));
  }

  /**
   * Handle chat request error
   */
  private handleChatError(error: unknown): never {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : '';

    this.outputChannel.appendLine(`ERROR: Chat request failed: ${errorMessage}`);
    if (errorStack) {
      this.outputChannel.appendLine(`Stack trace: ${errorStack}`);
    }

    const isToolError = errorMessage.includes('HarmonyError') || errorMessage.includes('unexpected tokens');

    if (isToolError) {
      this.outputChannel.appendLine('HINT: This appears to be a tool calling format error.');
      this.outputChannel.appendLine('The model may not support function calling properly.');
      this.outputChannel.appendLine('Try: 1) Using a different model, 2) Disabling tool calling in settings, or 3) Checking inference server logs');

      this.promptToolCallingError();
    } else {
      void vscode.window.showErrorMessage(
        `GitHub Copilot LLM Gateway: Chat request failed. ${errorMessage}`
      );
    }

    throw error;
  }

  private promptToolCallingError(): void {
    vscode.window
      .showErrorMessage(
        `GitHub Copilot LLM Gateway: Model failed to generate valid tool calls. This model may not support function calling. Check Output panel for details.`,
        'Open Output',
        'Disable Tool Calling'
      )
      .then(
        (selection: string | undefined) => {
          if (selection === 'Open Output') {
            this.outputChannel.show();
          } else if (selection === 'Disable Tool Calling') {
            void vscode.workspace
              .getConfiguration('github.copilot.llm-gateway')
              .update('enableToolCalling', false, vscode.ConfigurationTarget.Global);
          }
        },
        (err: unknown) => {
          this.outputChannel.appendLine(
            `Failed to show tool calling error prompt: ${err instanceof Error ? err.message : String(err)}`
          );
        }
      );
  }

  /**
   * Report a parser piece to the progress stream, updating stats.
   * `allowForceClose` is true only when flushing the parser — an 'E' piece mid-stream does
   * not count as a force-close.
   */
  private reportParserPiece(
    piece: ThinkingPiece,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    stats: StreamStats,
    allowForceClose: boolean
  ): void {
    if (piece.t === 'T') {
      stats.hadThinking = true;
      progress.report(new vscode.LanguageModelThinkingPart(piece.c));
    } else if (piece.t === 'E') {
      if (allowForceClose) {
        stats.thinkingForceClosed = true;
      }
      progress.report(new vscode.LanguageModelThinkingPart('', '', { vscode_reasoning_done: true }));
    } else if (piece.c) {
      stats.totalTextParts++;
      progress.report(new vscode.LanguageModelTextPart(piece.c));
    }
  }

  /**
   * Stream chunks from the inference server, reporting parts to VS Code.
   */
  private async streamResponse(
    requestOptions: OpenAIChatCompletionRequest,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken
  ): Promise<StreamStats> {
    const stats: StreamStats = {
      totalContent: '',
      totalToolCalls: 0,
      totalTextParts: 0,
      hadThinking: false,
      thinkingForceClosed: false,
    };
    const parser = new ThinkingParser();
    let inReasoningField = false;

    for await (const chunk of this.client.streamChatCompletion(requestOptions, token)) {
      if (token.isCancellationRequested) {
        break;
      }

      // reasoning_content field (LM Studio / DeepSeek-R1 structured output)
      if (chunk.reasoning_content) {
        stats.hadThinking = true;
        inReasoningField = true;
        progress.report(new vscode.LanguageModelThinkingPart(chunk.reasoning_content));
      }

      // text content — may contain raw <thinking> tags
      if (chunk.content) {
        if (inReasoningField) {
          inReasoningField = false;
          progress.report(new vscode.LanguageModelThinkingPart('', '', { vscode_reasoning_done: true }));
        }
        stats.totalContent += chunk.content;
        for (const piece of parser.process(chunk.content)) {
          this.reportParserPiece(piece, progress, stats, false);
        }
      }

      if (chunk.finished_tool_calls?.length) {
        for (const toolCall of chunk.finished_tool_calls) {
          stats.totalToolCalls++;
          this.processToolCall(toolCall, progress);
        }
      }
    }

    // Flush any remaining buffered content from the parser; 'E' here signals a force-close
    // (stream ended mid-think block — typically because token limit was reached while thinking).
    for (const piece of parser.flush()) {
      this.reportParserPiece(piece, progress, stats, true);
    }

    if (inReasoningField) {
      progress.report(new vscode.LanguageModelThinkingPart('', '', { vscode_reasoning_done: true }));
    }

    if (stats.thinkingForceClosed && stats.totalTextParts === 0 && stats.totalToolCalls === 0) {
      progress.report(new vscode.LanguageModelTextPart(
        '*(The model ran out of output tokens while thinking and could not produce a response. ' +
        'Try increasing the context length or max output tokens in LM Studio, ' +
        'or disable thinking for this model.)*'
      ));
    }

    return stats;
  }

  /**
   * Provide language model chat response - streams responses from inference server
   */
  async provideLanguageModelChatResponse(
    model: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken
  ): Promise<void> {
    this.outputChannel.appendLine(`Sending chat request to model: ${model.id}`);
    this.outputChannel.appendLine(`Tool mode: ${options.toolMode}, Tools: ${options.tools?.length || 0}`);
    this.outputChannel.appendLine(`Message count: ${messages.length}`);

    this.showWelcomeNotification(model.id);

    const openAIMessages: OpenAIRequestMessage[] = [];
    for (const msg of messages) {
      openAIMessages.push(...this.convertMessage(msg));
    }
    this.outputChannel.appendLine(`Converted to ${openAIMessages.length} OpenAI messages`);

    for (let i = 0; i < openAIMessages.length; i++) {
      const msg = openAIMessages[i];
      const toolCallId = typeof msg.tool_call_id === 'string' ? msg.tool_call_id : 'none';
      this.outputChannel.appendLine(
        `  Message ${i + 1}: role=${msg.role}, hasContent=${!!msg.content}, hasToolCalls=${!!msg.tool_calls}, toolCallId=${toolCallId}`
      );
    }

    const modelMaxContext = this.config.defaultMaxTokens || GatewayProvider.DEFAULT_CONTEXT_TOKENS;
    const desiredOutputTokens = Math.min(
      this.config.defaultMaxOutputTokens || GatewayProvider.DEFAULT_OUTPUT_TOKENS,
      Math.floor(modelMaxContext / 2)
    );
    const toolsTokenEstimate = options.tools
      ? Math.ceil(
          (JSON.stringify(options.tools).length / GatewayProvider.CHARS_PER_TOKEN) *
            GatewayProvider.INPUT_OVERHEAD_RATIO
        )
      : 0;
    const maxInputTokens =
      modelMaxContext - desiredOutputTokens - toolsTokenEstimate - GatewayProvider.CONTEXT_BUFFER_TOKENS;

    const truncatedMessages = this.truncateMessagesToFit(openAIMessages, maxInputTokens);
    if (truncatedMessages.length < openAIMessages.length) {
      this.outputChannel.appendLine(
        `WARNING: Truncated conversation from ${openAIMessages.length} to ${truncatedMessages.length} messages to fit context limit`
      );
    }

    const inputText = truncatedMessages
      .map((m) => {
        let text = typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '');
        if (m.tool_calls) {
          text += JSON.stringify(m.tool_calls);
        }
        return text;
      })
      .join('\n');

    const toolsOverhead = options.tools ? Math.ceil(JSON.stringify(options.tools).length / GatewayProvider.CHARS_PER_TOKEN) : 0;
    const estimatedInputTokens = await this.provideTokenCount(model, inputText, token);
    const safeMaxOutputTokens = this.calculateSafeMaxOutputTokens(estimatedInputTokens, toolsOverhead);

    this.outputChannel.appendLine(
      `Token estimate: input=${estimatedInputTokens}, tools=${toolsOverhead}, model_context=${modelMaxContext}, chosen_max_tokens=${safeMaxOutputTokens}`
    );

    const hasTools = this.config.enableToolCalling && options.tools && options.tools.length > 0;
    const temperature = hasTools ? (this.config.agentTemperature ?? 0) : GatewayProvider.DEFAULT_TEMPERATURE;

    const requestOptions: Record<string, unknown> = {
      model: model.id,
      messages: truncatedMessages,
      max_tokens: safeMaxOutputTokens,
      temperature,
    };

    const toolsConfig = this.buildToolsConfig(options);
    if (toolsConfig) {
      requestOptions.tools = toolsConfig;
      if (options.toolMode !== undefined) {
        requestOptions.tool_choice =
          options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
      }
      requestOptions.parallel_tool_calls = this.config.parallelToolCalling;
      this.outputChannel.appendLine(`Sending ${toolsConfig.length} tools to model (parallel: ${this.config.parallelToolCalling})`);
    }

    if (options.modelOptions) {
      Object.assign(requestOptions, options.modelOptions);
    }

    const debugRequest = JSON.stringify(requestOptions, null, 2);
    this.outputChannel.appendLine(
      debugRequest.length > 2000 ? `Request (truncated): ${debugRequest.substring(0, 2000)}...` : `Request: ${debugRequest}`
    );

    try {
      const stats = await this.streamResponse(
        requestOptions as unknown as OpenAIChatCompletionRequest,
        progress,
        token
      );

      this.outputChannel.appendLine(
        `Completed chat request, received ${stats.totalContent.length} chars, ${stats.totalTextParts} text parts, ${stats.totalToolCalls} tool calls`
      );

      if (
        stats.totalContent.length === 0 &&
        stats.totalToolCalls === 0 &&
        !stats.hadThinking &&
        !stats.thinkingForceClosed
      ) {
        const toolCount = requestOptions.tools ? (requestOptions.tools as unknown[]).length : 0;
        await this.handleEmptyResponse(model, inputText, openAIMessages.length, toolCount, token, progress);
      }
    } catch (error) {
      this.handleChatError(error);
    }
  }

  /**
   * Provide token count estimation. Uses a rough ~CHARS_PER_TOKEN-per-token approximation;
   * for real accuracy a tokenizer library would be needed.
   */
  async provideTokenCount(
    _model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatMessage,
    _token: vscode.CancellationToken
  ): Promise<number> {
    let content: string;

    if (typeof text === 'string') {
      content = text;
    } else {
      content = text.content
        .filter((part): part is vscode.LanguageModelTextPart => part instanceof vscode.LanguageModelTextPart)
        .map((part) => part.value)
        .join('');
    }

    return Math.ceil(content.length / GatewayProvider.CHARS_PER_TOKEN);
  }

  /**
   * Show a timed notification with a link to settings (once per session)
   */
  private showWelcomeNotification(modelId: string): void {
    if (this.hasShownWelcomeNotification) {
      return;
    }
    this.hasShownWelcomeNotification = true;

    void vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `LLM Gateway: ${modelId}  —  [Settings](command:workbench.action.openSettings?%22github.copilot.llm-gateway%22)`,
        cancellable: false,
      },
      () => new Promise<void>((resolve) => setTimeout(resolve, GatewayProvider.TOOL_RESPONSE_DELAY_MS))
    );
  }

  /**
   * Load configuration from VS Code settings
   */
  private loadConfig(): GatewayConfig {
    const config = vscode.workspace.getConfiguration('github.copilot.llm-gateway');

    const cfg: GatewayConfig = {
      serverUrl: config.get<string>('serverUrl', 'http://localhost:8000'),
      apiKey: config.get<string>('apiKey', ''),
      requestTimeout: config.get<number>('requestTimeout', GatewayProvider.DEFAULT_REQUEST_TIMEOUT_MS),
      defaultMaxTokens: config.get<number>('defaultMaxTokens', GatewayProvider.DEFAULT_CONTEXT_TOKENS),
      defaultMaxOutputTokens: config.get<number>('defaultMaxOutputTokens', GatewayProvider.FALLBACK_OUTPUT_TOKENS),
      enableImageInput: config.get<boolean>('enableImageInput', false),
      enableToolCalling: config.get<boolean>('enableToolCalling', true),
      parallelToolCalling: config.get<boolean>('parallelToolCalling', true),
      agentTemperature: config.get<number>('agentTemperature', 0),
    };

    if (cfg.requestTimeout <= 0) {
      this.outputChannel.appendLine(
        `ERROR: requestTimeout must be > 0; using default ${GatewayProvider.DEFAULT_REQUEST_TIMEOUT_MS}`
      );
      cfg.requestTimeout = GatewayProvider.DEFAULT_REQUEST_TIMEOUT_MS;
    }

    try {
      new URL(cfg.serverUrl);
    } catch {
      this.outputChannel.appendLine(`ERROR: Invalid server URL: ${cfg.serverUrl}`);
      throw new Error(`Invalid server URL: ${cfg.serverUrl}`);
    }

    if (cfg.defaultMaxOutputTokens >= cfg.defaultMaxTokens) {
      const adjusted = Math.max(
        GatewayProvider.MIN_OUTPUT_TOKENS,
        cfg.defaultMaxTokens - GatewayProvider.ADJUST_TOKEN_BUFFER
      );
      this.outputChannel.appendLine(
        `WARNING: github.copilot.llm-gateway.defaultMaxOutputTokens (${cfg.defaultMaxOutputTokens}) >= defaultMaxTokens (${cfg.defaultMaxTokens}). Adjusting to ${adjusted}.`
      );
      void vscode.window.showWarningMessage(
        `GitHub Copilot LLM Gateway: 'defaultMaxOutputTokens' was >= 'defaultMaxTokens'. Adjusted to ${adjusted} to avoid request errors.`
      );
      cfg.defaultMaxOutputTokens = adjusted;
    }

    return cfg;
  }

  /**
   * Reload configuration and update client
   */
  private reloadConfig(): void {
    this.config = this.loadConfig();
    this.client.updateConfig(this.config);
    this.outputChannel.appendLine('Configuration reloaded');
  }
}
