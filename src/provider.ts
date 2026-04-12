import * as vscode from 'vscode';
import { GatewayClient } from './client';
import { GatewayConfig, OpenAIChatCompletionRequest, OpenAIMessage } from './types';
import {
  convertMessage,
  NormalizedMessage,
  NormalizedPart,
  NormalizedRole,
} from './messageConverter';
import {
  TOKEN_CONSTANTS,
  buildInputText,
  calculateMaxInputTokens,
  calculateSafeMaxOutputTokens,
  estimateTextTokens,
  truncateMessagesToFit,
} from './tokenBudget';
import { tryRepairJson } from './jsonRepair';
import { fillMissingRequiredProperties } from './toolSchema';
import { buildChatRequest, OpenAIToolDefinition, ToolChoice } from './requestBuilder';
import {
  StreamChunk,
  StreamReporter,
  isEmptyStreamResult,
  streamResponse,
} from './responseStreamer';

const DEFAULT_REQUEST_TIMEOUT_MS = 60000;
const WELCOME_NOTIFICATION_DELAY_MS = 3000;
const DEFAULT_TEMPERATURE = 0.7;
const DEBUG_REQUEST_MAX_LOG_LENGTH = 2000;
const MAX_TOOL_ARGS_LOG_LENGTH = 1000;
const MAX_TOOL_DESCRIPTION_LOG_LENGTH = 100;

/**
 * Language model provider for OpenAI-compatible inference servers.
 *
 * This class is the VS Code surface area; most of the logic lives in focused
 * pure modules (messageConverter, tokenBudget, responseStreamer, etc.) which
 * are unit-tested independently.
 */
export class GatewayProvider implements vscode.LanguageModelChatProvider {
  private readonly client: GatewayClient;
  private config: GatewayConfig;
  private readonly outputChannel: vscode.OutputChannel;
  /** Per-request map from tool name → inputSchema, populated by buildToolsConfig. */
  private readonly currentToolSchemas: Map<string, Record<string, unknown> | undefined> = new Map();
  private hasShownWelcomeNotification = false;

  private readonly _onDidChangeLanguageModelChatInformation = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this._onDidChangeLanguageModelChatInformation.event;

  constructor(context: vscode.ExtensionContext) {
    this.outputChannel = vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
    this.config = this.loadConfig();
    this.client = new GatewayClient(this.config, (msg) => this.outputChannel.appendLine(msg));

    context.subscriptions.push(
      this._onDidChangeLanguageModelChatInformation,
      vscode.workspace.onDidChangeConfiguration((e: vscode.ConfigurationChangeEvent) => {
        if (e.affectsConfiguration('github.copilot.llm-gateway')) {
          this.outputChannel.appendLine('Configuration changed, reloading...');
          this.reloadConfig();
          this._onDidChangeLanguageModelChatInformation.fire();
        }
      })
    );
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

      const models = response.data.map(
        (model) => {
          const serverContext = model.max_model_len ?? model.context_length ?? model.context_window;
          const totalContext = serverContext ?? this.config.defaultMaxTokens;
          const maxOutputTokens = Math.min(
            this.config.defaultMaxOutputTokens,
            totalContext - TOKEN_CONSTANTS.ADJUST_TOKEN_BUFFER
          );
          const maxInputTokens = totalContext - maxOutputTokens;

          if (serverContext) {
            this.outputChannel.appendLine(
              `  Model ${model.id}: server-reported context ${serverContext} tokens (input=${maxInputTokens}, output=${maxOutputTokens})`
            );
          }

          return {
            id: model.id,
            name: model.id,
            family: 'llm-gateway',
            maxInputTokens,
            maxOutputTokens,
            version: '1.0.0',
            capabilities: {
              imageInput: this.config.enableImageInput,
              toolCalling: this.config.enableToolCalling,
            },
          } as vscode.LanguageModelChatInformation;
        }
      );

      this.outputChannel.appendLine(
        `Found ${models.length} models: ${models.map((m) => m.id).join(', ')}`
      );
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

    const openAIMessages = this.convertAllMessages(messages);
    this.outputChannel.appendLine(`Converted to ${openAIMessages.length} OpenAI messages`);
    this.logMessageStructure(openAIMessages);

    const modelMaxContext = (model.maxInputTokens + model.maxOutputTokens) || TOKEN_CONSTANTS.DEFAULT_CONTEXT_TOKENS;
    const configuredMaxOutput =
      model.maxOutputTokens || TOKEN_CONSTANTS.DEFAULT_OUTPUT_TOKENS;
    const toolsSerializedLength = options.tools ? JSON.stringify(options.tools).length : 0;

    const maxInputTokens = calculateMaxInputTokens({
      modelMaxContext,
      configuredMaxOutput,
      toolsSerializedLength,
    });

    const truncatedMessages = truncateMessagesToFit(openAIMessages, maxInputTokens, (msg) =>
      this.outputChannel.appendLine(msg)
    );
    if (truncatedMessages.length < openAIMessages.length) {
      this.outputChannel.appendLine(
        `WARNING: Truncated conversation from ${openAIMessages.length} to ${truncatedMessages.length} messages to fit context limit`
      );
    }

    const inputText = buildInputText(truncatedMessages);
    const toolsOverhead = Math.ceil(toolsSerializedLength / TOKEN_CONSTANTS.CHARS_PER_TOKEN);
    const estimatedInputTokens = await this.provideTokenCount(model, inputText, token);
    const safeMaxOutputTokens = calculateSafeMaxOutputTokens({
      estimatedInputTokens,
      toolsOverhead,
      modelMaxContext,
      configuredMaxOutput,
    });

    this.outputChannel.appendLine(
      `Token estimate: input=${estimatedInputTokens}, tools=${toolsOverhead}, model_context=${modelMaxContext}, chosen_max_tokens=${safeMaxOutputTokens}`
    );

    const tools = this.buildToolsConfig(options);
    const hasTools = tools !== undefined && tools.length > 0;
    const temperature = hasTools ? this.config.agentTemperature ?? 0 : DEFAULT_TEMPERATURE;

    const requestOptions = buildChatRequest({
      model: model.id,
      messages: truncatedMessages,
      maxTokens: safeMaxOutputTokens,
      temperature,
      tools,
      toolChoice: hasTools ? this.mapToolChoice(options.toolMode) : undefined,
      parallelToolCalls: hasTools ? this.config.parallelToolCalling : undefined,
      extraOptions: options.modelOptions,
    });

    if (hasTools) {
      this.outputChannel.appendLine(
        `Sending ${tools.length} tools to model (parallel: ${this.config.parallelToolCalling})`
      );
    }

    this.logRequest(requestOptions);

    try {
      const reporter = this.createStreamReporter(progress);
      const chunks = this.client.streamChatCompletion(requestOptions, token);
      const stats = await streamResponse({
        chunks: chunks as AsyncIterable<StreamChunk>,
        reporter,
        isCancelled: () => token.isCancellationRequested,
        resolveToolCallArgs: (toolCall) => this.resolveToolCallArgs(toolCall),
      });

      this.outputChannel.appendLine(
        `Completed chat request, received ${stats.totalContent.length} chars, ${stats.totalTextParts} text parts, ${stats.totalToolCalls} tool calls`
      );

      if (isEmptyStreamResult(stats)) {
        const toolCount = tools?.length ?? 0;
        await this.handleEmptyResponse(model, inputText, openAIMessages.length, toolCount, token, progress);
      }
    } catch (error) {
      this.handleChatError(error);
    }
  }

  /**
   * Provide token count estimation (rough char/4 approximation).
   */
  async provideTokenCount(
    _model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatMessage,
    _token: vscode.CancellationToken
  ): Promise<number> {
    if (typeof text === 'string') {
      return estimateTextTokens(text);
    }
    const textValue = text.content
      .filter((part): part is vscode.LanguageModelTextPart => part instanceof vscode.LanguageModelTextPart)
      .map((part) => part.value)
      .join('');
    return estimateTextTokens(textValue);
  }

  // ---------- message classification ----------

  private convertAllMessages(messages: readonly vscode.LanguageModelChatMessage[]): OpenAIMessage[] {
    const result: OpenAIMessage[] = [];
    const log = (msg: string): void => this.outputChannel.appendLine(msg);
    for (const msg of messages) {
      const normalized: NormalizedMessage = {
        role: this.mapRole(msg.role),
        parts: msg.content.map((part) => this.classifyPart(part)),
      };
      result.push(
        ...convertMessage(normalized, { enableImageInput: this.config.enableImageInput }, log)
      );
    }
    return result;
  }

  private mapRole(role: vscode.LanguageModelChatMessageRole): NormalizedRole {
    if (role === vscode.LanguageModelChatMessageRole.Assistant) {
      return 'assistant';
    }
    return 'user';
  }

  /**
   * Translate a vscode LanguageModel*Part into the plain data shape used by
   * messageConverter. Falls back to duck typing for older VS Code versions
   * where the constructors may not match.
   */
  private classifyPart(part: unknown): NormalizedPart {
    if (part instanceof vscode.LanguageModelTextPart) {
      return { kind: 'text', value: part.value };
    }
    if (part instanceof vscode.LanguageModelToolResultPart) {
      return {
        kind: 'toolResult',
        callId: part.callId,
        content: typeof part.content === 'string' ? part.content : JSON.stringify(part.content),
      };
    }
    if (part instanceof vscode.LanguageModelToolCallPart) {
      return {
        kind: 'toolCall',
        callId: part.callId,
        name: part.name,
        input: part.input,
      };
    }
    if (part instanceof vscode.LanguageModelDataPart) {
      return { kind: 'image', mimeType: part.mimeType, data: part.data };
    }
    return this.classifyPartDuckTyped(part);
  }

  private classifyPartDuckTyped(part: unknown): NormalizedPart {
    if (typeof part !== 'object' || part === null) {
      return { kind: 'unknown' };
    }
    const anyPart = part as Record<string, unknown>;

    if ('callId' in anyPart && 'content' in anyPart && !('name' in anyPart)) {
      this.outputChannel.appendLine(`  Found tool result (duck-typed): callId=${anyPart.callId}`);
      return {
        kind: 'toolResult',
        callId: String(anyPart.callId),
        content:
          typeof anyPart.content === 'string' ? anyPart.content : JSON.stringify(anyPart.content),
      };
    }
    if ('callId' in anyPart && 'name' in anyPart && 'input' in anyPart) {
      this.outputChannel.appendLine(
        `  Found tool call (duck-typed): callId=${anyPart.callId}, name=${anyPart.name}`
      );
      return {
        kind: 'toolCall',
        callId: String(anyPart.callId),
        name: String(anyPart.name),
        input: anyPart.input,
      };
    }
    return { kind: 'unknown' };
  }

  // ---------- tool config + stream adapters ----------

  private mapToolChoice(toolMode: vscode.LanguageModelChatToolMode | undefined): ToolChoice | undefined {
    if (toolMode === undefined) {
      return undefined;
    }
    return toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
  }

  private buildToolsConfig(
    options: vscode.ProvideLanguageModelChatResponseOptions
  ): OpenAIToolDefinition[] | undefined {
    if (!this.config.enableToolCalling || !options.tools || options.tools.length === 0) {
      return undefined;
    }

    this.currentToolSchemas.clear();

    return options.tools.map((tool) => {
      this.outputChannel.appendLine(`Tool: ${tool.name}`);
      this.outputChannel.appendLine(
        `  Description: ${tool.description?.substring(0, MAX_TOOL_DESCRIPTION_LOG_LENGTH) || 'none'}...`
      );

      const schema = tool.inputSchema as Record<string, unknown> | undefined;
      this.currentToolSchemas.set(tool.name, schema);

      if (schema?.required && Array.isArray(schema.required)) {
        this.outputChannel.appendLine(
          `  Required properties: ${(schema.required as string[]).join(', ')}`
        );
      }

      return {
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
        },
      };
    });
  }

  /**
   * Parse and patch tool call arguments before reporting them upstream.
   * Used as the `resolveToolCallArgs` callback on the response streamer.
   */
  private resolveToolCallArgs(toolCall: {
    id: string;
    name: string;
    arguments: string;
  }): Record<string, unknown> {
    this.outputChannel.appendLine(`\n=== TOOL CALL RECEIVED ===`);
    this.outputChannel.appendLine(`  ID: ${toolCall.id}`);
    this.outputChannel.appendLine(`  Name: ${toolCall.name}`);
    this.outputChannel.appendLine(
      `  Raw arguments: ${toolCall.arguments.substring(0, MAX_TOOL_ARGS_LOG_LENGTH)}${
        toolCall.arguments.length > MAX_TOOL_ARGS_LOG_LENGTH ? '...' : ''
      }`
    );

    const log = (msg: string): void => this.outputChannel.appendLine(msg);
    let args = tryRepairJson(toolCall.arguments, log) as Record<string, unknown> | null;

    if (args === null) {
      this.outputChannel.appendLine(`  ERROR: Failed to parse tool call arguments`);
      this.outputChannel.appendLine(`  Full arguments: ${toolCall.arguments}`);
      args = {};
    } else {
      const argKeys = Object.keys(args);
      this.outputChannel.appendLine(
        `  Parsed argument keys: ${argKeys.length > 0 ? argKeys.join(', ') : '(none)'}`
      );
    }

    const toolSchema = this.currentToolSchemas.get(toolCall.name);
    if (toolSchema) {
      args = fillMissingRequiredProperties(args, toolSchema, log);
    }

    this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
    return args;
  }

  private createStreamReporter(
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): StreamReporter {
    return {
      reportText: (text) => progress.report(new vscode.LanguageModelTextPart(text)),
      reportThinking: (text) => progress.report(new vscode.LanguageModelThinkingPart(text)),
      reportThinkingDone: () =>
        progress.report(new vscode.LanguageModelThinkingPart('', '', { vscode_reasoning_done: true })),
      reportToolCall: (id, name, args) =>
        progress.report(new vscode.LanguageModelToolCallPart(id, name, args)),
    };
  }

  // ---------- logging helpers ----------

  private logMessageStructure(openAIMessages: readonly OpenAIMessage[]): void {
    for (let i = 0; i < openAIMessages.length; i++) {
      const msg = openAIMessages[i];
      const toolCallId = typeof msg.tool_call_id === 'string' ? msg.tool_call_id : 'none';
      this.outputChannel.appendLine(
        `  Message ${i + 1}: role=${msg.role}, hasContent=${!!msg.content}, hasToolCalls=${!!msg.tool_calls}, toolCallId=${toolCallId}`
      );
    }
  }

  private logRequest(request: OpenAIChatCompletionRequest): void {
    const debugRequest = JSON.stringify(request, null, 2);
    this.outputChannel.appendLine(
      debugRequest.length > DEBUG_REQUEST_MAX_LOG_LENGTH
        ? `Request (truncated): ${debugRequest.substring(0, DEBUG_REQUEST_MAX_LOG_LENGTH)}...`
        : `Request: ${debugRequest}`
    );
  }

  // ---------- error / UI helpers ----------

  private async handleEmptyResponse(
    model: vscode.LanguageModelChatInformation,
    inputText: string,
    messageCount: number,
    toolCount: number,
    token: vscode.CancellationToken,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): Promise<void> {
    const inputTokenCount = await this.provideTokenCount(model, inputText, token);
    const modelMaxContext = (model.maxInputTokens + model.maxOutputTokens) || TOKEN_CONSTANTS.DEFAULT_CONTEXT_TOKENS;

    this.outputChannel.appendLine(`WARNING: Model returned empty response with no tool calls.`);
    this.outputChannel.appendLine(`  Input tokens estimated: ${inputTokenCount}`);
    this.outputChannel.appendLine(`  Messages in conversation: ${messageCount}`);
    this.outputChannel.appendLine(`  Tools provided: ${toolCount}`);

    const errorHint =
      toolCount > 0
        ? `The model returned an empty response. This typically indicates the model failed to generate valid output with tool calling enabled. Check the inference server logs for errors.`
        : `The model returned an empty response. Check the inference server logs for details.`;

    this.outputChannel.appendLine(`  Issue: ${errorHint}`);

    const errorMessage =
      `I was unable to generate a response. ${errorHint}\n\n` +
      `Diagnostic info:\n- Model: ${model.id}\n- Tools provided: ${toolCount}\n` +
      `- Estimated input tokens: ${inputTokenCount}\n- Context limit: ${modelMaxContext}\n\n` +
      `Check the "GitHub Copilot LLM Gateway" output panel for detailed logs.`;

    progress.report(new vscode.LanguageModelTextPart(errorMessage));
  }

  private handleChatError(error: unknown): never {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : '';

    this.outputChannel.appendLine(`ERROR: Chat request failed: ${errorMessage}`);
    if (errorStack) {
      this.outputChannel.appendLine(`Stack trace: ${errorStack}`);
    }

    const isToolError =
      errorMessage.includes('HarmonyError') || errorMessage.includes('unexpected tokens');

    if (isToolError) {
      this.outputChannel.appendLine('HINT: This appears to be a tool calling format error.');
      this.outputChannel.appendLine('The model may not support function calling properly.');
      this.outputChannel.appendLine(
        'Try: 1) Using a different model, 2) Disabling tool calling in settings, or 3) Checking inference server logs'
      );
      this.promptToolCallingError();
    } else {
      vscode.window.showErrorMessage(
        `GitHub Copilot LLM Gateway: Chat request failed. ${errorMessage}`
      );
    }

    throw error;
  }

  private promptOpenSettings(message: string): void {
    vscode.window.showErrorMessage(message, 'Open Settings').then(
      (selection: string | undefined) => {
        if (selection === 'Open Settings') {
          vscode.commands.executeCommand(
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
            vscode.workspace
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

  private showWelcomeNotification(modelId: string): void {
    if (this.hasShownWelcomeNotification) {
      return;
    }
    this.hasShownWelcomeNotification = true;

    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `LLM Gateway: ${modelId}  —  [Settings](command:workbench.action.openSettings?%22github.copilot.llm-gateway%22)`,
        cancellable: false,
      },
      () => new Promise<void>((resolve) => setTimeout(resolve, WELCOME_NOTIFICATION_DELAY_MS))
    );
  }

  // ---------- config ----------

  private loadConfig(): GatewayConfig {
    const config = vscode.workspace.getConfiguration('github.copilot.llm-gateway');

    const cfg: GatewayConfig = {
      serverUrl: config.get<string>('serverUrl', 'http://localhost:8000'),
      apiKey: config.get<string>('apiKey', ''),
      requestTimeout: config.get<number>('requestTimeout', DEFAULT_REQUEST_TIMEOUT_MS),
      defaultMaxTokens: config.get<number>('defaultMaxTokens', TOKEN_CONSTANTS.DEFAULT_CONTEXT_TOKENS),
      defaultMaxOutputTokens: config.get<number>(
        'defaultMaxOutputTokens',
        TOKEN_CONSTANTS.FALLBACK_OUTPUT_TOKENS
      ),
      enableImageInput: config.get<boolean>('enableImageInput', false),
      enableToolCalling: config.get<boolean>('enableToolCalling', true),
      parallelToolCalling: config.get<boolean>('parallelToolCalling', true),
      agentTemperature: config.get<number>('agentTemperature', 0),
    };

    if (cfg.requestTimeout <= 0) {
      this.outputChannel.appendLine(
        `ERROR: requestTimeout must be > 0; using default ${DEFAULT_REQUEST_TIMEOUT_MS}`
      );
      cfg.requestTimeout = DEFAULT_REQUEST_TIMEOUT_MS;
    }

    try {
      new URL(cfg.serverUrl);
    } catch {
      this.outputChannel.appendLine(`ERROR: Invalid server URL: ${cfg.serverUrl}`);
      throw new Error(`Invalid server URL: ${cfg.serverUrl}`);
    }

    if (cfg.defaultMaxOutputTokens >= cfg.defaultMaxTokens) {
      const adjusted = Math.max(
        TOKEN_CONSTANTS.MIN_OUTPUT_TOKENS,
        cfg.defaultMaxTokens - TOKEN_CONSTANTS.ADJUST_TOKEN_BUFFER
      );
      this.outputChannel.appendLine(
        `WARNING: github.copilot.llm-gateway.defaultMaxOutputTokens (${cfg.defaultMaxOutputTokens}) >= defaultMaxTokens (${cfg.defaultMaxTokens}). Adjusting to ${adjusted}.`
      );
      vscode.window.showWarningMessage(
        `GitHub Copilot LLM Gateway: 'defaultMaxOutputTokens' was >= 'defaultMaxTokens'. Adjusted to ${adjusted} to avoid request errors.`
      );
      cfg.defaultMaxOutputTokens = adjusted;
    }

    return cfg;
  }

  private reloadConfig(): void {
    this.config = this.loadConfig();
    this.client.updateConfig(this.config);
    this.outputChannel.appendLine('Configuration reloaded');
  }
}
