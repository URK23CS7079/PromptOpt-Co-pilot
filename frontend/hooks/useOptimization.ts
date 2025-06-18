import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import { apiClient } from '../utils/api';
import {
  OptimizationConfig,
  OptimizationResult,
  OptimizationPhase,
  OptimizationStatus,
  WebSocketMessage,
  ErrorType
} from '../utils/types';

/**
 * Configuration options for the useOptimization hook
 */
export interface OptimizationHookOptions {
  /** Enable automatic reconnection for WebSocket */
  autoReconnect?: boolean;
  /** Maximum number of reconnection attempts */
  maxReconnectAttempts?: number;
  /** Interval between reconnection attempts in milliseconds */
  reconnectInterval?: number;
  /** Enable progress polling fallback when WebSocket is unavailable */
  enablePolling?: boolean;
  /** Polling interval in milliseconds */
  pollingInterval?: number;
  /** Maximum number of optimizations to keep in history */
  maxHistorySize?: number;
  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Real-time progress information for an optimization run
 */
export interface OptimizationProgress {
  /** Current phase of optimization */
  phase: OptimizationPhase;
  /** Progress percentage (0-100) */
  percentage: number;
  /** Current step description */
  currentStep: string;
  /** Estimated time remaining in milliseconds */
  estimatedTimeRemaining?: number;
  /** Resource usage information */
  resourceUsage: {
    cpuUsage: number;
    memoryUsage: number;
    tokensProcessed: number;
    tokensRemaining: number;
  };
  /** Detailed phase breakdown */
  phaseDetails: {
    variantGeneration?: {
      completed: number;
      total: number;
      currentVariant?: string;
    };
    evaluation?: {
      completed: number;
      total: number;
      currentMetric?: string;
    };
    analysis?: {
      completed: number;
      total: number;
      currentAnalysis?: string;
    };
  };
}

/**
 * Complete optimization run data
 */
export interface OptimizationRun {
  /** Unique identifier for the optimization run */
  id: string;
  /** Configuration used for this optimization */
  config: OptimizationConfig;
  /** Current status of the optimization */
  status: OptimizationStatus;
  /** Progress information */
  progress: OptimizationProgress;
  /** Start time */
  startTime: Date;
  /** End time (if completed) */
  endTime?: Date;
  /** Optimization results (if completed) */
  result?: OptimizationResult;
  /** Error information (if failed) */
  error?: {
    type: ErrorType;
    message: string;
    details?: any;
    retryable: boolean;
  };
}

/**
 * WebSocket connection status
 */
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';

/**
 * Hook state interface
 */
export interface OptimizationState {
  /** Current active optimization */
  currentOptimization: OptimizationRun | null;
  /** History of completed optimizations */
  optimizationHistory: OptimizationRun[];
  /** Loading state */
  isLoading: boolean;
  /** Error state */
  error: string | null;
  /** WebSocket connection status */
  connectionStatus: ConnectionStatus;
  /** Timestamp of last update */
  lastUpdate: Date | null;
  /** Retry count for current operation */
  retryCount: number;
}

/**
 * Action types for state management
 */
type OptimizationAction =
  | { type: 'START_OPTIMIZATION'; payload: { run: OptimizationRun } }
  | { type: 'UPDATE_PROGRESS'; payload: { progress: OptimizationProgress } }
  | { type: 'COMPLETE_OPTIMIZATION'; payload: { result: OptimizationResult } }
  | { type: 'FAIL_OPTIMIZATION'; payload: { error: OptimizationRun['error'] } }
  | { type: 'STOP_OPTIMIZATION' }
  | { type: 'SET_LOADING'; payload: { isLoading: boolean } }
  | { type: 'SET_ERROR'; payload: { error: string | null } }
  | { type: 'CLEAR_ERROR' }
  | { type: 'SET_CONNECTION_STATUS'; payload: { status: ConnectionStatus } }
  | { type: 'UPDATE_LAST_UPDATE' }
  | { type: 'INCREMENT_RETRY_COUNT' }
  | { type: 'RESET_RETRY_COUNT' }
  | { type: 'ADD_TO_HISTORY'; payload: { run: OptimizationRun } }
  | { type: 'CLEAR_HISTORY' };

/**
 * Initial state for the optimization hook
 */
const initialState: OptimizationState = {
  currentOptimization: null,
  optimizationHistory: [],
  isLoading: false,
  error: null,
  connectionStatus: 'disconnected',
  lastUpdate: null,
  retryCount: 0,
};

/**
 * Reducer function for optimization state management
 */
function optimizationReducer(state: OptimizationState, action: OptimizationAction): OptimizationState {
  switch (action.type) {
    case 'START_OPTIMIZATION':
      return {
        ...state,
        currentOptimization: action.payload.run,
        isLoading: true,
        error: null,
        retryCount: 0,
        lastUpdate: new Date(),
      };

    case 'UPDATE_PROGRESS':
      if (!state.currentOptimization) return state;
      return {
        ...state,
        currentOptimization: {
          ...state.currentOptimization,
          progress: action.payload.progress,
        },
        lastUpdate: new Date(),
      };

    case 'COMPLETE_OPTIMIZATION':
      if (!state.currentOptimization) return state;
      const completedRun = {
        ...state.currentOptimization,
        status: 'completed' as OptimizationStatus,
        result: action.payload.result,
        endTime: new Date(),
      };
      return {
        ...state,
        currentOptimization: null,
        optimizationHistory: [completedRun, ...state.optimizationHistory].slice(0, 10), // Keep last 10
        isLoading: false,
        lastUpdate: new Date(),
      };

    case 'FAIL_OPTIMIZATION':
      if (!state.currentOptimization) return state;
      const failedRun = {
        ...state.currentOptimization,
        status: 'failed' as OptimizationStatus,
        error: action.payload.error,
        endTime: new Date(),
      };
      return {
        ...state,
        currentOptimization: null,
        optimizationHistory: [failedRun, ...state.optimizationHistory].slice(0, 10),
        isLoading: false,
        error: action.payload.error?.message || 'Optimization failed',
        lastUpdate: new Date(),
      };

    case 'STOP_OPTIMIZATION':
      if (!state.currentOptimization) return state;
      const stoppedRun = {
        ...state.currentOptimization,
        status: 'cancelled' as OptimizationStatus,
        endTime: new Date(),
      };
      return {
        ...state,
        currentOptimization: null,
        optimizationHistory: [stoppedRun, ...state.optimizationHistory].slice(0, 10),
        isLoading: false,
        lastUpdate: new Date(),
      };

    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload.isLoading,
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload.error,
        isLoading: false,
      };

    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
        retryCount: 0,
      };

    case 'SET_CONNECTION_STATUS':
      return {
        ...state,
        connectionStatus: action.payload.status,
      };

    case 'UPDATE_LAST_UPDATE':
      return {
        ...state,
        lastUpdate: new Date(),
      };

    case 'INCREMENT_RETRY_COUNT':
      return {
        ...state,
        retryCount: state.retryCount + 1,
      };

    case 'RESET_RETRY_COUNT':
      return {
        ...state,
        retryCount: 0,
      };

    case 'ADD_TO_HISTORY':
      return {
        ...state,
        optimizationHistory: [action.payload.run, ...state.optimizationHistory].slice(0, 10),
      };

    case 'CLEAR_HISTORY':
      return {
        ...state,
        optimizationHistory: [],
      };

    default:
      return state;
  }
}

/**
 * Return type for the useOptimization hook
 */
export interface UseOptimizationReturn {
  // State
  currentOptimization: OptimizationRun | null;
  optimizationHistory: OptimizationRun[];
  isLoading: boolean;
  error: string | null;
  progress: OptimizationProgress | null;
  connectionStatus: ConnectionStatus;
  lastUpdate: Date | null;

  // Actions
  startOptimization: (config: OptimizationConfig) => Promise<void>;
  stopOptimization: () => Promise<void>;
  retryOptimization: () => Promise<void>;
  clearError: () => void;
  clearHistory: () => void;

  // Computed values
  canRetry: boolean;
  isConnected: boolean;
  hasActiveOptimization: boolean;
}

/**
 * Custom hook for managing optimization workflow state and WebSocket connections
 * 
 * @param options - Configuration options for the hook
 * @returns Object containing state, actions, and computed values
 * 
 * @example
 * ```tsx
 * const {
 *   currentOptimization,
 *   isLoading,
 *   error,
 *   progress,
 *   startOptimization,
 *   stopOptimization,
 *   connectionStatus
 * } = useOptimization({
 *   autoReconnect: true,
 *   enablePolling: true
 * });
 * 
 * // Start optimization
 * await startOptimization({
 *   prompt: "Original prompt",
 *   targetMetrics: ["accuracy", "latency"],
 *   maxVariants: 10
 * });
 * ```
 */
export function useOptimization(options: OptimizationHookOptions = {}): UseOptimizationReturn {
  const {
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectInterval = 3000,
    enablePolling = true,
    pollingInterval = 2000,
    maxHistorySize = 10,
    debug = false,
  } = options;

  const [state, dispatch] = useReducer(optimizationReducer, initialState);
  
  // Refs for managing WebSocket and timers
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  /**
   * Log debug messages if debug mode is enabled
   */
  const debugLog = useCallback((message: string, data?: any) => {
    if (debug) {
      console.log(`[useOptimization] ${message}`, data);
    }
  }, [debug]);

  /**
   * Create and manage WebSocket connection
   */
  const createWebSocketConnection = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      debugLog('WebSocket already connected');
      return;
    }

    debugLog('Creating WebSocket connection');
    dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'connecting' } });

    try {
      const wsUrl = `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000'}/ws/optimization`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        debugLog('WebSocket connected');
        dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'connected' } });
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          debugLog('Received WebSocket message', message);
          
          switch (message.type) {
            case 'optimization_progress':
              dispatch({ 
                type: 'UPDATE_PROGRESS', 
                payload: { progress: message.data as OptimizationProgress } 
              });
              break;
            
            case 'optimization_complete':
              dispatch({ 
                type: 'COMPLETE_OPTIMIZATION', 
                payload: { result: message.data as OptimizationResult } 
              });
              break;
            
            case 'optimization_error':
              dispatch({ 
                type: 'FAIL_OPTIMIZATION', 
                payload: { error: message.data as OptimizationRun['error'] } 
              });
              break;
            
            default:
              debugLog('Unknown message type', message.type);
          }
        } catch (error) {
          debugLog('Error parsing WebSocket message', error);
        }
      };

      ws.onerror = (error) => {
        debugLog('WebSocket error', error);
        dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'error' } });
      };

      ws.onclose = (event) => {
        debugLog('WebSocket closed', { code: event.code, reason: event.reason });
        dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'disconnected' } });
        
        // Attempt reconnection if enabled and within retry limits
        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'reconnecting' } });
          
          reconnectTimeoutRef.current = setTimeout(() => {
            debugLog(`Reconnection attempt ${reconnectAttemptsRef.current}`);
            createWebSocketConnection();
          }, reconnectInterval);
        }
      };

    } catch (error) {
      debugLog('Error creating WebSocket connection', error);
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'error' } });
    }
  }, [autoReconnect, maxReconnectAttempts, reconnectInterval, debugLog]);

  /**
   * Close WebSocket connection and cleanup
   */
  const closeWebSocketConnection = useCallback(() => {
    if (wsRef.current) {
      debugLog('Closing WebSocket connection');
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    dispatch({ type: 'SET_CONNECTION_STATUS', payload: { status: 'disconnected' } });
  }, [debugLog]);

  /**
   * Start polling for optimization progress (fallback when WebSocket unavailable)
   */
  const startPolling = useCallback((optimizationId: string) => {
    if (!enablePolling || pollingIntervalRef.current) return;

    debugLog('Starting progress polling', { optimizationId });
    
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const progress = await apiClient.getOptimizationProgress(optimizationId);
        dispatch({ type: 'UPDATE_PROGRESS', payload: { progress } });
        
        if (progress.percentage === 100) {
          const result = await apiClient.getOptimizationResult(optimizationId);
          dispatch({ type: 'COMPLETE_OPTIMIZATION', payload: { result } });
          stopPolling();
        }
      } catch (error) {
        debugLog('Polling error', error);
        // Continue polling unless it's a critical error
      }
    }, pollingInterval);
  }, [enablePolling, pollingInterval, debugLog]);

  /**
   * Stop polling for optimization progress
   */
  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      debugLog('Stopping progress polling');
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, [debugLog]);

  /**
   * Start a new optimization run
   */
  const startOptimization = useCallback(async (config: OptimizationConfig) => {
    try {
      debugLog('Starting optimization', config);
      dispatch({ type: 'SET_LOADING', payload: { isLoading: true } });
      dispatch({ type: 'CLEAR_ERROR' });

      const response = await apiClient.startOptimization(config);
      
      const optimizationRun: OptimizationRun = {
        id: response.id,
        config,
        status: 'running',
        progress: {
          phase: 'variant_generation',
          percentage: 0,
          currentStep: 'Initializing optimization...',
          resourceUsage: {
            cpuUsage: 0,
            memoryUsage: 0,
            tokensProcessed: 0,
            tokensRemaining: 0,
          },
          phaseDetails: {},
        },
        startTime: new Date(),
      };

      dispatch({ type: 'START_OPTIMIZATION', payload: { run: optimizationRun } });

      // Start progress tracking
      if (state.connectionStatus === 'connected') {
        // WebSocket will handle updates
        debugLog('Using WebSocket for progress updates');
      } else {
        // Fallback to polling
        debugLog('Using polling for progress updates');
        startPolling(response.id);
      }

    } catch (error: any) {
      debugLog('Error starting optimization', error);
      dispatch({ 
        type: 'SET_ERROR', 
        payload: { error: error.message || 'Failed to start optimization' } 
      });
    }
  }, [state.connectionStatus, startPolling, debugLog]);

  /**
   * Stop the current optimization run
   */
  const stopOptimization = useCallback(async () => {
    if (!state.currentOptimization) return;

    try {
      debugLog('Stopping optimization', { id: state.currentOptimization.id });
      await apiClient.stopOptimization(state.currentOptimization.id);
      dispatch({ type: 'STOP_OPTIMIZATION' });
      stopPolling();
    } catch (error: any) {
      debugLog('Error stopping optimization', error);
      dispatch({ 
        type: 'SET_ERROR', 
        payload: { error: error.message || 'Failed to stop optimization' } 
      });
    }
  }, [state.currentOptimization, stopPolling, debugLog]);

  /**
   * Retry the last failed optimization
   */
  const retryOptimization = useCallback(async () => {
    const lastRun = state.optimizationHistory[0];
    if (!lastRun || !lastRun.error?.retryable) return;

    debugLog('Retrying optimization', { id: lastRun.id });
    dispatch({ type: 'INCREMENT_RETRY_COUNT' });
    await startOptimization(lastRun.config);
  }, [state.optimizationHistory, startOptimization, debugLog]);

  /**
   * Clear the current error state
   */
  const clearError = useCallback(() => {
    debugLog('Clearing error');
    dispatch({ type: 'CLEAR_ERROR' });
  }, [debugLog]);

  /**
   * Clear optimization history
   */
  const clearHistory = useCallback(() => {
    debugLog('Clearing history');
    dispatch({ type: 'CLEAR_HISTORY' });
  }, [debugLog]);

  // Computed values
  const canRetry = useMemo(() => {
    const lastRun = state.optimizationHistory[0];
    return lastRun?.error?.retryable === true && state.retryCount < 3;
  }, [state.optimizationHistory, state.retryCount]);

  const isConnected = useMemo(() => {
    return state.connectionStatus === 'connected';
  }, [state.connectionStatus]);

  const hasActiveOptimization = useMemo(() => {
    return state.currentOptimization !== null;
  }, [state.currentOptimization]);

  const progress = useMemo(() => {
    return state.currentOptimization?.progress || null;
  }, [state.currentOptimization]);

  // Initialize WebSocket connection on mount
  useEffect(() => {
    createWebSocketConnection();
    
    return () => {
      closeWebSocketConnection();
      stopPolling();
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      closeWebSocketConnection();
      stopPolling();
    };
  }, [closeWebSocketConnection, stopPolling]);

  return {
    // State
    currentOptimization: state.currentOptimization,
    optimizationHistory: state.optimizationHistory,
    isLoading: state.isLoading,
    error: state.error,
    progress,
    connectionStatus: state.connectionStatus,
    lastUpdate: state.lastUpdate,

    // Actions
    startOptimization,
    stopOptimization,
    retryOptimization,
    clearError,
    clearHistory,

    // Computed values
    canRetry,
    isConnected,
    hasActiveOptimization,
  };
}