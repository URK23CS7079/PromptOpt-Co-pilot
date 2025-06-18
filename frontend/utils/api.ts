import { 
  ApiResponse, 
  ApiError as ApiErrorInterface,
  Prompt,
  OptimizationJob,
  Evaluation,
  Dataset,
  SystemHealth,
  PaginatedResponse,
  CreatePromptRequest,
  UpdatePromptRequest,
  CreateOptimizationRequest,
  CreateEvaluationRequest,
  CreateDatasetRequest,
  UpdateDatasetRequest
} from './types';

/**
 * Configuration options for the API client
 */
export interface ApiClientOptions {
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Base delay for exponential backoff in milliseconds (default: 1000) */
  retryDelay?: number;
  /** Enable request/response caching (default: true) */
  enableCache?: boolean;
  /** Default cache TTL in milliseconds (default: 300000 - 5 minutes) */
  cacheTtl?: number;
  /** Custom headers to include with all requests */
  defaultHeaders?: Record<string, string>;
  /** Enable request/response logging (default: false in production) */
  enableLogging?: boolean;
}

/**
 * Options for individual API requests
 */
export interface RequestOptions {
  /** Override default timeout for this request */
  timeout?: number;
  /** Custom headers for this request */
  headers?: Record<string, string>;
  /** Skip cache for this request */
  skipCache?: boolean;
  /** Custom cache TTL for this request */
  cacheTtl?: number;
  /** AbortController signal for request cancellation */
  signal?: AbortSignal;
  /** Skip automatic retry for this request */
  skipRetry?: boolean;
}

/**
 * Progress callback for file uploads
 */
export type ProgressCallback = (progress: number) => void;

/**
 * Cache entry structure
 */
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

/**
 * WebSocket event types
 */
export interface WebSocketEvents {
  'optimization:progress': { jobId: string; progress: number; status: string };
  'optimization:complete': { jobId: string; result: any };
  'optimization:error': { jobId: string; error: string };
  'evaluation:complete': { evaluationId: string; result: any };
  'system:health': SystemHealth;
  'connection:established': void;
  'connection:lost': void;
  'connection:restored': void;
}

/**
 * WebSocket message structure
 */
interface WebSocketMessage {
  type: keyof WebSocketEvents;
  data: any;
  timestamp: number;
}

/**
 * Custom API Error class with detailed information
 */
export class ApiError extends Error implements ApiErrorInterface {
  public readonly status: number;
  public readonly code: string;
  public readonly details?: any;
  public readonly timestamp: Date;
  public readonly requestId?: string;

  constructor(
    message: string,
    status: number = 500,
    code: string = 'UNKNOWN_ERROR',
    details?: any,
    requestId?: string
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
    this.timestamp = new Date();
    this.requestId = requestId;

    // Maintain proper stack trace for debugging
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }
  }

  /**
   * Create user-friendly error message based on status code
   */
  getUserFriendlyMessage(): string {
    switch (this.status) {
      case 400:
        return 'Invalid request. Please check your input and try again.';
      case 401:
        return 'Authentication required. Please log in and try again.';
      case 403:
        return 'Access denied. You do not have permission to perform this action.';
      case 404:
        return 'The requested resource was not found.';
      case 409:
        return 'Conflict detected. The resource already exists or has been modified.';
      case 422:
        return 'Invalid data provided. Please correct the errors and try again.';
      case 429:
        return 'Too many requests. Please wait a moment and try again.';
      case 500:
        return 'Server error occurred. Please try again later.';
      case 502:
      case 503:
      case 504:
        return 'Service temporarily unavailable. Please try again later.';
      default:
        return this.message || 'An unexpected error occurred.';
    }
  }

  /**
   * Convert error to JSON for logging
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      status: this.status,
      code: this.code,
      details: this.details,
      timestamp: this.timestamp,
      requestId: this.requestId,
      stack: this.stack
    };
  }
}

/**
 * WebSocket Manager for real-time communications
 */
export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private protocols?: string | string[];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private eventListeners = new Map<keyof WebSocketEvents, Set<Function>>();
  private isConnecting = false;
  private shouldReconnect = true;

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;
    this.shouldReconnect = true;

    try {
      this.ws = new WebSocket(this.url, this.protocols);
      
      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);

      // Wait for connection to establish
      await new Promise<void>((resolve, reject) => {
        if (!this.ws) return reject(new Error('WebSocket not initialized'));
        
        const openHandler = () => {
          this.ws?.removeEventListener('open', openHandler);
          this.ws?.removeEventListener('error', errorHandler);
          resolve();
        };
        
        const errorHandler = (event: Event) => {
          this.ws?.removeEventListener('open', openHandler);
          this.ws?.removeEventListener('error', errorHandler);
          reject(new Error('WebSocket connection failed'));
        };

        this.ws.addEventListener('open', openHandler);
        this.ws.addEventListener('error', errorHandler);
      });
    } catch (error) {
      this.isConnecting = false;
      throw error;
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.shouldReconnect = false;
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Intentional disconnect');
      this.ws = null;
    }
  }

  /**
   * Send message to WebSocket server
   */
  send<K extends keyof WebSocketEvents>(type: K, data: WebSocketEvents[K]): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected');
    }

    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now()
    };

    this.ws.send(JSON.stringify(message));
  }

  /**
   * Add event listener for WebSocket events
   */
  on<K extends keyof WebSocketEvents>(
    event: K,
    listener: (data: WebSocketEvents[K]) => void
  ): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(listener);
  }

  /**
   * Remove event listener
   */
  off<K extends keyof WebSocketEvents>(
    event: K,
    listener: (data: WebSocketEvents[K]) => void
  ): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
  }

  /**
   * Get current connection state
   */
  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  /**
   * Check if WebSocket is connected
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private handleOpen(): void {
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.emit('connection:established', undefined);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.emit(message.type, message.data);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private handleClose(event: CloseEvent): void {
    this.isConnecting = false;
    this.ws = null;

    if (event.code !== 1000 && this.shouldReconnect) {
      this.emit('connection:lost', undefined);
      this.scheduleReconnect();
    }
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.isConnecting = false;
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
        this.emit('connection:restored', undefined);
      } catch (error) {
        console.error('Reconnection failed:', error);
        this.scheduleReconnect();
      }
    }, delay);
  }

  private emit<K extends keyof WebSocketEvents>(event: K, data: WebSocketEvents[K]): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in WebSocket event listener for ${event}:`, error);
        }
      });
    }
  }
}

/**
 * Main API Client class for HTTP communications
 */
export class ApiClient {
  private baseUrl: string;
  private options: Required<ApiClientOptions>;
  private cache = new Map<string, CacheEntry<any>>();
  private pendingRequests = new Map<string, Promise<any>>();

  constructor(baseUrl: string, options: ApiClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.options = {
      timeout: options.timeout ?? 30000,
      maxRetries: options.maxRetries ?? 3,
      retryDelay: options.retryDelay ?? 1000,
      enableCache: options.enableCache ?? true,
      cacheTtl: options.cacheTtl ?? 300000, // 5 minutes
      defaultHeaders: options.defaultHeaders ?? {},
      enableLogging: options.enableLogging ?? (process.env.NODE_ENV === 'development')
    };
  }

  /**
   * Perform GET request
   */
  async get<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    return this.request<T>('GET', endpoint, undefined, options);
  }

  /**
   * Perform POST request
   */
  async post<T>(endpoint: string, data: any, options: RequestOptions = {}): Promise<T> {
    return this.request<T>('POST', endpoint, data, options);
  }

  /**
   * Perform PUT request
   */
  async put<T>(endpoint: string, data: any, options: RequestOptions = {}): Promise<T> {
    return this.request<T>('PUT', endpoint, data, options);
  }

  /**
   * Perform DELETE request
   */
  async delete<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    return this.request<T>('DELETE', endpoint, undefined, options);
  }

  /**
   * Upload file with progress tracking
   */
  async upload(
    endpoint: string, 
    file: File, 
    onProgress?: ProgressCallback,
    options: RequestOptions = {}
  ): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;
    const formData = new FormData();
    formData.append('file', file);

    // Create AbortController for cancellation
    const controller = new AbortController();
    const signal = options.signal ?? controller.signal;

    const headers = {
      ...this.options.defaultHeaders,
      ...options.headers
    };

    // Remove Content-Type header to let browser set it with boundary
    delete headers['Content-Type'];

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData,
        signal
      });

      if (!response.ok) {
        const errorData = await this.parseErrorResponse(response);
        throw new ApiError(
          errorData.message || 'Upload failed',
          response.status,
          errorData.code,
          errorData.details,
          response.headers.get('X-Request-ID') || undefined
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw this.handleRequestError(error);
    }
  }

  /**
   * Core request method with retry logic and caching
   */
  private async request<T>(
    method: string,
    endpoint: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const cacheKey = `${method}:${url}:${JSON.stringify(data)}`;

    // Check cache for GET requests
    if (method === 'GET' && this.options.enableCache && !options.skipCache) {
      const cached = this.getFromCache<T>(cacheKey);
      if (cached) return cached;

      // Check for pending identical requests
      const pending = this.pendingRequests.get(cacheKey);
      if (pending) return pending as Promise<T>;
    }

    const requestPromise = this.executeRequest<T>(method, url, data, options);

    // Store pending request for deduplication
    if (method === 'GET') {
      this.pendingRequests.set(cacheKey, requestPromise);
    }

    try {
      const result = await requestPromise;

      // Cache successful GET requests
      if (method === 'GET' && this.options.enableCache && !options.skipCache) {
        this.setCache(cacheKey, result, options.cacheTtl ?? this.options.cacheTtl);
      }

      return result;
    } finally {
      // Clean up pending request
      this.pendingRequests.delete(cacheKey);
    }
  }

  /**
   * Execute HTTP request with retry logic
   */
  private async executeRequest<T>(
    method: string,
    url: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    const maxRetries = options.skipRetry ? 0 : this.options.maxRetries;
    let lastError: Error;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.performRequest<T>(method, url, data, options);
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors (4xx) except 429 (rate limit)
        if (error instanceof ApiError && error.status >= 400 && error.status < 500 && error.status !== 429) {
          throw error;
        }

        // Don't retry on the last attempt
        if (attempt === maxRetries) {
          throw error;
        }

        // Wait before retrying with exponential backoff
        const delay = this.options.retryDelay * Math.pow(2, attempt);
        await this.sleep(delay);

        if (this.options.enableLogging) {
          console.warn(`API request failed, retrying (${attempt + 1}/${maxRetries + 1}):`, error);
        }
      }
    }

    throw lastError!;
  }

  /**
   * Perform single HTTP request
   */
  private async performRequest<T>(
    method: string,
    url: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout ?? this.options.timeout);

    // Combine signals if provided
    if (options.signal) {
      options.signal.addEventListener('abort', () => controller.abort());
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.options.defaultHeaders,
      ...options.headers
    };

    // Add CSRF token if available
    const csrfToken = this.getCSRFToken();
    if (csrfToken) {
      headers['X-CSRF-Token'] = csrfToken;
    }

    const requestInit: RequestInit = {
      method,
      headers,
      signal: controller.signal
    };

    if (data && method !== 'GET') {
      requestInit.body = JSON.stringify(data);
    }

    if (this.options.enableLogging) {
      console.log(`API Request: ${method} ${url}`, { data, headers: options.headers });
    }

    try {
      const response = await fetch(url, requestInit);
      clearTimeout(timeoutId);

      if (this.options.enableLogging) {
        console.log(`API Response: ${method} ${url}`, {
          status: response.status,
          statusText: response.statusText
        });
      }

      if (!response.ok) {
        const errorData = await this.parseErrorResponse(response);
        throw new ApiError(
          errorData.message || `HTTP ${response.status} ${response.statusText}`,
          response.status,
          errorData.code || 'HTTP_ERROR',
          errorData.details,
          response.headers.get('X-Request-ID') || undefined
        );
      }

      // Handle empty responses
      const contentType = response.headers.get('Content-Type');
      if (!contentType || !contentType.includes('application/json')) {
        return undefined as unknown as T;
      }

      const result = await response.json();
      return result as T;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof ApiError) {
        throw error;
      }

      throw this.handleRequestError(error);
    }
  }

  /**
   * Parse error response from server
   */
  private async parseErrorResponse(response: Response): Promise<any> {
    try {
      const contentType = response.headers.get('Content-Type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
    } catch {
      // Fall back to status text if JSON parsing fails
    }

    return {
      message: response.statusText || 'Unknown error',
      code: 'HTTP_ERROR',
      status: response.status
    };
  }

  /**
   * Handle request errors (network, timeout, etc.)
   */
  private handleRequestError(error: any): ApiError {
    if (error.name === 'AbortError') {
      return new ApiError('Request timeout', 408, 'TIMEOUT');
    }

    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      return new ApiError('Network error - please check your connection', 0, 'NETWORK_ERROR');
    }

    return new ApiError(
      error.message || 'Request failed',
      0,
      'REQUEST_ERROR',
      error
    );
  }

  /**
   * Get CSRF token from meta tag or cookie
   */
  private getCSRFToken(): string | null {
    // Try to get from meta tag first
    const metaTag = document.querySelector('meta[name="csrf-token"]');
    if (metaTag) {
      return metaTag.getAttribute('content');
    }

    // Try to get from cookie
    const match = document.cookie.match(/csrftoken=([^;]+)/);
    return match ? match[1] : null;
  }

  /**
   * Get cached response
   */
  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  /**
   * Set cache entry
   */
  private setCache<T>(key: string, data: T, ttl: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });

    // Limit cache size (remove oldest entries)
    if (this.cache.size > 1000) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    this.pendingRequests.clear();
  }

  /**
   * Utility method to sleep for given milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize API client with environment-specific configuration
const getApiBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    // Client-side
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
  }
  // Server-side
  return process.env.API_URL || 'http://localhost:8000/api/v1';
};

const getWebSocketUrl = (): string => {
  if (typeof window !== 'undefined') {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'ws://localhost:8000';
    return baseUrl.replace(/^http/, 'ws') + '/ws';
  }
  return 'ws://localhost:8000/ws';
};

// Export default instances
export const apiClient = new ApiClient(getApiBaseUrl(), {
  enableLogging: process.env.NODE_ENV === 'development',
  defaultHeaders: {
    'X-Client-Version': '1.0.0'
  }
});

export const wsManager = new WebSocketManager(getWebSocketUrl());

/**
 * Prompt API functions
 */
export const promptApi = {
  /**
   * Get all prompts with pagination
   */
  async getAll(page = 1, limit = 20, search?: string): Promise<PaginatedResponse<Prompt>> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(search && { search })
    });
    return apiClient.get<PaginatedResponse<Prompt>>(`/prompts?${params}`);
  },

  /**
   * Get prompt by ID
   */
  async getById(id: string): Promise<Prompt> {
    return apiClient.get<Prompt>(`/prompts/${id}`);
  },

  /**
   * Create new prompt
   */
  async create(data: CreatePromptRequest): Promise<Prompt> {
    return apiClient.post<Prompt>('/prompts', data);
  },

  /**
   * Update existing prompt
   */
  async update(id: string, data: UpdatePromptRequest): Promise<Prompt> {
    return apiClient.put<Prompt>(`/prompts/${id}`, data);
  },

  /**
   * Delete prompt
   */
  async delete(id: string): Promise<void> {
    return apiClient.delete<void>(`/prompts/${id}`);
  },

  /**
   * Duplicate prompt
   */
  async duplicate(id: string): Promise<Prompt> {
    return apiClient.post<Prompt>(`/prompts/${id}/duplicate`);
  }
};

/**
 * Optimization API functions
 */
export const optimizationApi = {
  /**
   * Start optimization job
   */
  async start(data: CreateOptimizationRequest): Promise<OptimizationJob> {
    return apiClient.post<OptimizationJob>('/optimizations', data);
  },

  /**
   * Get optimization job status
   */
  async getStatus(jobId: string): Promise<OptimizationJob> {
    return apiClient.get<OptimizationJob>(`/optimizations/${jobId}`);
  },

  /**
   * Cancel optimization job
   */
  async cancel(jobId: string): Promise<void> {
    return apiClient.post<void>(`/optimizations/${jobId}/cancel`);
  },

  /**
   * Get optimization history
   */
  async getHistory(page = 1, limit = 20): Promise<PaginatedResponse<OptimizationJob>> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString()
    });
    return apiClient.get<PaginatedResponse<OptimizationJob>>(`/optimizations?${params}`);
  }
};

/**
 * Evaluation API functions
 */
export const evaluationApi = {
  /**
   * Create evaluation
   */
  async create(data: CreateEvaluationRequest): Promise<Evaluation> {
    return apiClient.post<Evaluation>('/evaluations', data);
  },

  /**
   * Get evaluation results
   */
  async getResults(id: string): Promise<Evaluation> {
    return apiClient.get<Evaluation>(`/evaluations/${id}`);
  },

  /**
   * Get evaluation history
   */
  async getHistory(page = 1, limit = 20): Promise<PaginatedResponse<Evaluation>> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString()
    });
    return apiClient.get<PaginatedResponse<Evaluation>>(`/evaluations?${params}`);
  }
};

/**
 * Dataset API functions
 */
export const datasetApi = {
  /**
   * Get all datasets
   */
  async getAll(): Promise<Dataset[]> {
    return apiClient.get<Dataset[]>('/datasets');
  },

  /**
   * Get dataset by ID
   */
  async getById(id: string): Promise<Dataset> {
    return apiClient.get<Dataset>(`/datasets/${id}`);
  },

  /**
   * Create dataset
   */
  async create(data: CreateDatasetRequest): Promise<Dataset> {
    return apiClient.post<Dataset>('/datasets', data);
  },

  /**
   * Update dataset
   */
  async update(id: string, data: UpdateDatasetRequest): Promise<Dataset> {
    return apiClient.put<Dataset>(`/datasets/${id}`, data);
  },

  /**
   * Delete dataset
   */
  async delete(id: string): Promise<void> {
    return apiClient.delete<void>(`/datasets/${id}`);
  },

  /**
   * Upload dataset file
   */
  async upload(file: File, onProgress?: ProgressCallback): Promise<Dataset> {
    return apiClient.upload('/datasets/upload', file, onProgress);
  }
};

/**
 * System API functions
 */
export const systemApi = {
  /**
   * Get system health status
   */
  async getHealth(): Promise<SystemHealth> {
    return apiClient.get<SystemHealth>('/system/health', { skipCache: true });
  },

  /**
   * Get system metrics
   */
  async getMetrics(): Promise<any> {
    return apiClient.get<any>('/system/metrics', { skipCache: true });
  }
};

/**
 * Utility functions for API error handling
 */
export const apiUtils = {
  /**
   * Check if error is a network error
   */
  isNetworkError(error: unknown): boolean {
    return error instanceof ApiError && error.code === 'NETWORK_ERROR';
  },

  /**
   * Check if error is a timeout error
   */
  isTimeoutError(error: unknown): boolean {
    return error instanceof ApiError && error.code === 'TIMEOUT';
  },

  /**
   * Check if error is a server error (5xx)
   */
  isServerError(error: unknown): boolean {
    return error instanceof ApiError && error.status >= 500;
  },

  /**
   * Check if error is a client error (4xx)
   */
  isClientError(error: unknown): boolean {
    return error instanceof ApiError && error.status >= 400 && error.status < 500;
  },

  /**
   * Get user-friendly error message
   */
  getErrorMessage(error: unknown): string {
    if (error instanceof ApiError) {
      return error.getUserFriendlyMessage();
    }
    return 'An unexpected error occurred';
  },

  /**
   * Format error for logging
   */
  formatErrorForLogging(error: unknown): object {
    if (error instanceof ApiError) {
      return error.toJSON();
    }
    return {
      name: 'Unknown Error',
      message: String(error),
      timestamp: new Date()
    };
  }
};

/**
 * API configuration for different environments
 */
export const apiConfig = {
  /**
   * Development configuration
   */
  development: {
    timeout: 60000, // Longer timeout for development
    maxRetries: 2,
    retryDelay: 500,
    enableCache: false, // Disable cache in development
    enableLogging: true,
    cacheTtl: 60000 // 1 minute cache
  },

  /**
   * Production configuration
   */
  production: {
    timeout: 30000,
    maxRetries: 3,
    retryDelay: 1000,
    enableCache: true,
    enableLogging: false,
    cacheTtl: 300000 // 5 minutes cache
  },

  /**
   * Test configuration
   */
  test: {
    timeout: 5000,
    maxRetries: 0,
    retryDelay: 0,
    enableCache: false,
    enableLogging: false,
    cacheTtl: 0
  }
};

/**
 * Create API client with environment-specific configuration
 */
export function createApiClient(environment: keyof typeof apiConfig = 'production'): ApiClient {
  const config = apiConfig[environment];
  return new ApiClient(getApiBaseUrl(), config);
}

/**
 * React hook for API error handling
 * Usage: const { handleError, error, clearError } = useApiError();
 */
export function useApiError() {
  const [error, setError] = useState<ApiError | null>(null);

  const handleError = useCallback((err: unknown) => {
    if (err instanceof ApiError) {
      setError(err);
    } else {
      setError(new ApiError(
        'An unexpected error occurred',
        0,
        'UNKNOWN_ERROR',
        err
      ));
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return { error, handleError, clearError };
}

/**
 * React hook for WebSocket connection management
 * Usage: const { isConnected, connect, disconnect } = useWebSocket();
 */
export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const handleConnectionEstablished = () => setIsConnected(true);
    const handleConnectionLost = () => setIsConnected(false);
    const handleConnectionRestored = () => setIsConnected(true);

    wsManager.on('connection:established', handleConnectionEstablished);
    wsManager.on('connection:lost', handleConnectionLost);
    wsManager.on('connection:restored', handleConnectionRestored);

    return () => {
      wsManager.off('connection:established', handleConnectionEstablished);
      wsManager.off('connection:lost', handleConnectionLost);
      wsManager.off('connection:restored', handleConnectionRestored);
    };
  }, []);

  const connect = useCallback(async () => {
    try {
      await wsManager.connect();
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      throw error;
    }
  }, []);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, []);

  return {
    isConnected,
    connect,
    disconnect,
    manager: wsManager
  };
}

/**
 * React hook for optimization job monitoring
 * Usage: const { job, startOptimization, cancelOptimization } = useOptimization();
 */
export function useOptimization() {
  const [job, setJob] = useState<OptimizationJob | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const startOptimization = useCallback(async (request: CreateOptimizationRequest) => {
    setIsLoading(true);
    try {
      const newJob = await optimizationApi.start(request);
      setJob(newJob);

      // Listen for progress updates
      const handleProgress = (data: WebSocketEvents['optimization:progress']) => {
        if (data.jobId === newJob.id) {
          setJob(prev => prev ? { ...prev, progress: data.progress, status: data.status } : null);
        }
      };

      const handleComplete = (data: WebSocketEvents['optimization:complete']) => {
        if (data.jobId === newJob.id) {
          setJob(prev => prev ? { ...prev, status: 'completed', result: data.result } : null);
        }
      };

      const handleError = (data: WebSocketEvents['optimization:error']) => {
        if (data.jobId === newJob.id) {
          setJob(prev => prev ? { ...prev, status: 'failed', error: data.error } : null);
        }
      };

      wsManager.on('optimization:progress', handleProgress);
      wsManager.on('optimization:complete', handleComplete);
      wsManager.on('optimization:error', handleError);

      return newJob;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const cancelOptimization = useCallback(async (jobId: string) => {
    await optimizationApi.cancel(jobId);
    setJob(prev => prev ? { ...prev, status: 'cancelled' } : null);
  }, []);

  return {
    job,
    isLoading,
    startOptimization,
    cancelOptimization
  };
}

/**
 * Request interceptor for adding authentication headers
 */
export function addAuthInterceptor(token: string) {
  apiClient.options.defaultHeaders['Authorization'] = `Bearer ${token}`;
}

/**
 * Remove authentication headers
 */
export function removeAuthInterceptor() {
  delete apiClient.options.defaultHeaders['Authorization'];
}

/**
 * Global error handler for unhandled API errors
 */
export function setupGlobalErrorHandler() {
  if (typeof window !== 'undefined') {
    window.addEventListener('unhandledrejection', (event) => {
      if (event.reason instanceof ApiError) {
        console.error('Unhandled API Error:', event.reason.toJSON());
        
        // You can integrate with error reporting service here
        // Example: Sentry.captureException(event.reason);
      }
    });
  }
}

// Export types for external use
export type { 
  ApiClientOptions, 
  RequestOptions, 
  ProgressCallback, 
  WebSocketEvents,
  WebSocketMessage 
};