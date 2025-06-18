import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Download,
  Filter,
  Search,
  RefreshCw,
  Eye,
  Clock,
  Zap,
  Target,
  Activity,
  AlertCircle,
  CheckCircle,
  ArrowUpRight,
  ArrowDownRight,
  MoreVertical,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

// TypeScript Interfaces
interface OptimizationResult {
  id: string;
  promptId: string;
  variantId: string;
  variantName: string;
  promptText: string;
  timestamp: string;
  metrics: {
    accuracy: number;
    latency: number;
    cost: number;
    toxicity: number;
    coherence: number;
    relevance: number;
    customMetrics?: Record<string, number>;
  };
  metadata: {
    model: string;
    temperature: number;
    maxTokens: number;
    testCases: number;
    iterations: number;
  };
  status: 'completed' | 'running' | 'failed';
  improvement: number;
  isBaseline: boolean;
}

interface OptimizationRun {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  results: OptimizationResult[];
  config: {
    objective: string;
    maxIterations: number;
    timeout: number;
  };
  progress: {
    current: number;
    total: number;
    currentVariant: string;
  };
}

interface ChartConfig {
  type: 'line' | 'bar' | 'scatter' | 'pie';
  metric: string;
  timeRange: string;
  groupBy: string;
}

interface FilterState {
  search: string;
  status: string[];
  dateRange: { start: string; end: string };
  metrics: { min: number; max: number; metric: string }[];
}

interface ExportOptions {
  format: 'csv' | 'json' | 'png';
  data: 'results' | 'charts' | 'summary';
  includeMetadata: boolean;
}

// API and Hook Imports (mocked for this example)
const useOptimization = () => ({
  currentRun: null as OptimizationRun | null,
  isRunning: false,
  progress: 0,
  subscribe: (callback: (data: any) => void) => {},
  unsubscribe: () => {}
});

const api = {
  getOptimizationResults: async (runId?: string): Promise<OptimizationResult[]> => {
    // Mock implementation
    return [];
  },
  getOptimizationRuns: async (): Promise<OptimizationRun[]> => {
    // Mock implementation
    return [];
  },
  exportResults: async (options: ExportOptions): Promise<Blob> => {
    // Mock implementation
    return new Blob();
  }
};

/**
 * MetricCard Component
 * Displays individual metric with trend indicators and comparison
 */
interface MetricCardProps {
  title: string;
  value: number | string;
  change?: number;
  changeType?: 'positive' | 'negative' | 'neutral';
  format?: 'number' | 'percentage' | 'currency' | 'time';
  subtitle?: string;
  icon?: React.ReactNode;
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changeType = 'neutral',
  format = 'number',
  subtitle,
  icon,
  loading = false
}) => {
  const formatValue = (val: number | string) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'percentage':
        return `${(val * 100).toFixed(1)}%`;
      case 'currency':
        return `$${val.toFixed(2)}`;
      case 'time':
        return `${val.toFixed(0)}ms`;
      default:
        return val.toFixed(2);
    }
  };

  const getTrendIcon = () => {
    if (!change) return null;
    
    if (changeType === 'positive') {
      return change > 0 ? <ArrowUpRight className="h-4 w-4 text-green-500" /> : <ArrowDownRight className="h-4 w-4 text-red-500" />;
    } else if (changeType === 'negative') {
      return change > 0 ? <ArrowDownRight className="h-4 w-4 text-red-500" /> : <ArrowUpRight className="h-4 w-4 text-green-500" />;
    }
    return null;
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            {icon && <div className="text-gray-500 dark:text-gray-400">{icon}</div>}
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</h3>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            {formatValue(value)}
          </div>
          {subtitle && (
            <p className="text-sm text-gray-500 dark:text-gray-400">{subtitle}</p>
          )}
        </div>
        {change !== undefined && (
          <div className="flex items-center gap-1 text-sm">
            {getTrendIcon()}
            <span className={`font-medium ${
              changeType === 'positive' && change > 0 ? 'text-green-600' :
              changeType === 'positive' && change < 0 ? 'text-red-600' :
              changeType === 'negative' && change > 0 ? 'text-red-600' :
              changeType === 'negative' && change < 0 ? 'text-green-600' :
              'text-gray-600'
            }`}>
              {Math.abs(change).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * VariantComparisonTable Component
 * Displays optimization results in a sortable, filterable table
 */
interface VariantComparisonTableProps {
  results: OptimizationResult[];
  onVariantSelect: (result: OptimizationResult) => void;
  loading?: boolean;
}

const VariantComparisonTable: React.FC<VariantComparisonTableProps> = ({
  results,
  onVariantSelect,
  loading = false
}) => {
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({
    key: 'metrics.accuracy',
    direction: 'desc'
  });
  const [filter, setFilter] = useState('');

  const sortedResults = useMemo(() => {
    const filtered = results.filter(result =>
      result.variantName.toLowerCase().includes(filter.toLowerCase()) ||
      result.promptText.toLowerCase().includes(filter.toLowerCase())
    );

    return [...filtered].sort((a, b) => {
      const aValue = sortConfig.key.split('.').reduce((obj, key) => obj?.[key], a as any);
      const bValue = sortConfig.key.split('.').reduce((obj, key) => obj?.[key], b as any);
      
      if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [results, sortConfig, filter]);

  const handleSort = (key: string) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const getSortIcon = (key: string) => {
    if (sortConfig.key !== key) return <ChevronDown className="h-4 w-4 text-gray-400" />;
    return sortConfig.direction === 'asc' ? 
      <ChevronUp className="h-4 w-4 text-blue-500" /> : 
      <ChevronDown className="h-4 w-4 text-blue-500" />;
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      completed: { color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200', icon: CheckCircle },
      running: { color: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200', icon: Activity },
      failed: { color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200', icon: AlertCircle }
    };

    const config = statusConfig[status as keyof typeof statusConfig];
    const Icon = config.icon;

    return (
      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${config.color}`}>
        <Icon className="h-3 w-3" />
        {status}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
        </div>
        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="p-4 animate-pulse">
              <div className="grid grid-cols-6 gap-4">
                {[...Array(6)].map((_, j) => (
                  <div key={j} className="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Variant Comparison</h3>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search variants..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('variantName')}
              >
                <div className="flex items-center gap-1">
                  Variant
                  {getSortIcon('variantName')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('metrics.accuracy')}
              >
                <div className="flex items-center gap-1">
                  Accuracy
                  {getSortIcon('metrics.accuracy')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('metrics.latency')}
              >
                <div className="flex items-center gap-1">
                  Latency
                  {getSortIcon('metrics.latency')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('metrics.cost')}
              >
                <div className="flex items-center gap-1">
                  Cost
                  {getSortIcon('metrics.cost')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('improvement')}
              >
                <div className="flex items-center gap-1">
                  Improvement
                  {getSortIcon('improvement')}
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {sortedResults.map((result) => (
              <tr 
                key={result.id}
                className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                onClick={() => onVariantSelect(result)}
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {result.variantName}
                        {result.isBaseline && (
                          <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                            Baseline
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
                        {result.promptText.substring(0, 50)}...
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900 dark:text-white">
                    {(result.metrics.accuracy * 100).toFixed(1)}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900 dark:text-white">
                    {result.metrics.latency.toFixed(0)}ms
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900 dark:text-white">
                    ${result.metrics.cost.toFixed(4)}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className={`text-sm font-medium ${
                    result.improvement > 0 ? 'text-green-600' : 
                    result.improvement < 0 ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {result.improvement > 0 ? '+' : ''}{result.improvement.toFixed(1)}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getStatusBadge(result.status)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onVariantSelect(result);
                    }}
                    className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/**
 * PerformanceChart Component
 * Renders various chart types for performance visualization
 */
interface PerformanceChartProps {
  data: any[];
  config: ChartConfig;
  title: string;
  loading?: boolean;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  data,
  config,
  title,
  loading = false
}) => {
  const chartColors = [
    '#3B82F6', '#EF4444', '#10B981', '#F59E0B',
    '#8B5CF6', '#F97316', '#06B6D4', '#84CC16'
  ];

  const renderChart = () => {
    switch (config.type) {
      case 'line':
        return (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis 
              dataKey="timestamp" 
              stroke="#6B7280"
              fontSize={12}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis stroke="#6B7280" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey={config.metric}
              stroke={chartColors[0]}
              strokeWidth={2}
              dot={{ fill: chartColors[0], strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: chartColors[0], strokeWidth: 2 }}
            />
          </LineChart>
        );

      case 'bar':
        return (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis dataKey="name" stroke="#6B7280" fontSize={12} />
            <YAxis stroke="#6B7280" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
            />
            <Legend />
            <Bar dataKey={config.metric} fill={chartColors[0]} radius={[4, 4, 0, 0]} />
          </BarChart>
        );

      case 'scatter':
        return (
          <ScatterChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis dataKey="x" stroke="#6B7280" fontSize={12} />
            <YAxis dataKey="y" stroke="#6B7280" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
            />
            <Scatter dataKey="value" fill={chartColors[0]} />
          </ScatterChart>
        );

      case 'pie':
        return (
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        );

      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4 animate-pulse"></div>
        <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">{title}</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {renderChart()}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

/**
 * OptimizationProgress Component
 * Shows real-time progress of optimization runs
 */
interface OptimizationProgressProps {
  run: OptimizationRun | null;
  onStop?: () => void;
}

const OptimizationProgress: React.FC<OptimizationProgressProps> = ({ run, onStop }) => {
  if (!run || run.status !== 'running') return null;

  const progress = (run.progress.current / run.progress.total) * 100;

  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-600 animate-pulse" />
          <h3 className="text-sm font-medium text-blue-900 dark:text-blue-100">
            Optimization in Progress: {run.name}
          </h3>
        </div>
        {onStop && (
          <button
            onClick={onStop}
            className="px-3 py-1 text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200"
          >
            Stop
          </button>
        )}
      </div>
      
      <div className="space-y-2">
        <div className="flex justify-between text-sm text-blue-700 dark:text-blue-300">
          <span>Progress: {run.progress.current} / {run.progress.total}</span>
          <span>{progress.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-sm text-blue-600 dark:text-blue-400">
          Current: {run.progress.currentVariant}
        </div>
      </div>
    </div>
  );
};

/**
 * ExportDialog Component
 * Handles data export functionality
 */
interface ExportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onExport: (options: ExportOptions) => void;
  loading?: boolean;
}

const ExportDialog: React.FC<ExportDialogProps> = ({ isOpen, onClose, onExport, loading = false }) => {
  const [options, setOptions] = useState<ExportOptions>({
    format: 'csv',
    data: 'results',
    includeMetadata: true
  });

  const handleExport = () => {
    onExport(options);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Export Results</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            ×
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Format
            </label>
            <select
              value={options.format}
              onChange={(e) => setOptions(prev => ({ ...prev, format: e.target.value as any }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
              <option value="png">PNG (Charts)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Data Type
            </label>
            <select
              value={options.data}
              onChange={(e) => setOptions(prev => ({ ...prev, data: e.target.value as any }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="results">Results</option>
              <option value="charts">Charts</option>
              <option value="summary">Summary</option>
            </select>
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="includeMetadata"
              checked={options.includeMetadata}
              onChange={(e) => setOptions(prev => ({ ...prev, includeMetadata: e.target.checked }))}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="includeMetadata" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
              Include metadata
            </label>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={loading}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-4 w-4" />
                Export
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Main ResultsDashboard Component
 * Orchestrates all sub-components and manages dashboard state
 */
const ResultsDashboard: React.FC = () => {
  // State management
  const [results, setResults] = useState<OptimizationResult[]>([]);
  const [runs, setRuns] = useState<OptimizationRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [selectedVariant, setSelectedVariant] = useState<OptimizationResult | null>(null);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    status: [],
    dateRange: { start: '', end: '' },
    metrics: []
  });
  const [chartConfigs, setChartConfigs] = useState<ChartConfig[]>([
    { type: 'line', metric: 'accuracy', timeRange: '7d', groupBy: 'variant' },
    { type: 'bar', metric: 'latency', timeRange: '7d', groupBy: 'variant' },
    { type: 'scatter', metric: 'cost', timeRange: '7d', groupBy: 'variant' }
  ]);

  // Hooks
  const { currentRun, isRunning, subscribe, unsubscribe } = useOptimization();

  // Data fetching
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [resultsData, runsData] = await Promise.all([
        api.getOptimizationResults(selectedRun || undefined),
        api.getOptimizationRuns()
      ]);
      
      setResults(resultsData);
      setRuns(runsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  }, [selectedRun]);

  // Effects
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (isRunning) {
      const handleUpdate = (data: any) => {
        // Update results with real-time data
        setResults(prev => [...prev, data]);
      };

      subscribe(handleUpdate);
      return () => unsubscribe();
    }
  }, [isRunning, subscribe, unsubscribe]);

  // Computed values
  const summary = useMemo(() => {
    if (!results.length) return null;

    const completed = results.filter(r => r.status === 'completed');
    const bestVariant = completed.reduce((best, current) => 
      current.metrics.accuracy > best.metrics.accuracy ? current : best
    , completed[0]);

    const avgImprovement = completed.reduce((sum, result) => sum + result.improvement, 0) / completed.length;
    const avgLatency = completed.reduce((sum, result) => sum + result.metrics.latency, 0) / completed.length;
    const totalCost = completed.reduce((sum, result) => sum + result.metrics.cost, 0);

    return {
      bestVariant: bestVariant?.variantName || 'N/A',
      avgImprovement,
      avgLatency,
      totalCost,
      completedTests: completed.length,
      successRate: (completed.length / results.length) * 100
    };
  }, [results]);

  // Chart data preparation
  const chartData = useMemo(() => {
    return chartConfigs.map(config => {
      const data = results.map(result => ({
        name: result.variantName,
        value: result.metrics[config.metric as keyof typeof result.metrics],
        timestamp: result.timestamp,
        ...result.metrics
      }));

      return {
        config,
        data,
        title: `${config.metric.charAt(0).toUpperCase() + config.metric.slice(1)} Performance`
      };
    });
  }, [results, chartConfigs]);

  // Event handlers
  const handleVariantSelect = (result: OptimizationResult) => {
    setSelectedVariant(result);
  };

  const handleExport = async (options: ExportOptions) => {
    try {
      setExportLoading(true);
      const blob = await api.exportResults(options);
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `optimization-results.${options.format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setShowExportDialog(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setExportLoading(false);
    }
  };

  const handleRetry = () => {
    fetchData();
  };

  // Error state
  if (error && !loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
          <div className="flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
            <div>
              <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
                Error loading results
              </h3>
              <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
            </div>
          </div>
          <button
            onClick={handleRetry}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Optimization Results
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Analyze and compare prompt optimization performance
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <select
            value={selectedRun || ''}
            onChange={(e) => setSelectedRun(e.target.value || null)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="">All Runs</option>
            {runs.map(run => (
              <option key={run.id} value={run.id}>
                {run.name}
              </option>
            ))}
          </select>
          
          <button
            onClick={() => setShowExportDialog(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
          
          <button
            onClick={fetchData}
            disabled={loading}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Optimization Progress */}
      <OptimizationProgress run={currentRun} />

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Best Variant"
            value={summary.bestVariant}
            icon={<Target className="h-5 w-5" />}
            loading={loading}
          />
          <MetricCard
            title="Average Improvement"
            value={summary.avgImprovement}
            format="percentage"
            changeType="positive"
            icon={<TrendingUp className="h-5 w-5" />}
            loading={loading}
          />
          <MetricCard
            title="Average Latency"
            value={summary.avgLatency}
            format="time"
            changeType="negative"
            icon={<Clock className="h-5 w-5" />}
            loading={loading}
          />
          <MetricCard
            title="Total Cost"
            value={summary.totalCost}
            format="currency"
            subtitle={`${summary.completedTests} tests completed`}
            icon={<Zap className="h-5 w-5" />}
            loading={loading}
          />
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {chartData.map((chart, index) => (
          <PerformanceChart
            key={index}
            data={chart.data}
            config={chart.config}
            title={chart.title}
            loading={loading}
          />
        ))}
      </div>

      {/* Variant Comparison Table */}
      <VariantComparisonTable
        results={results}
        onVariantSelect={handleVariantSelect}
        loading={loading}
      />

      {/* Export Dialog */}
      <ExportDialog
        isOpen={showExportDialog}
        onClose={() => setShowExportDialog(false)}
        onExport={handleExport}
        loading={exportLoading}
      />

      {/* Variant Detail Modal */}
      {selectedVariant && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Variant Details: {selectedVariant.variantName}
              </h3>
              <button
                onClick={() => setSelectedVariant(null)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                ×
              </button>
            </div>
            
            <div className="space-y-6">
              {/* Prompt Text */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Prompt Text
                </h4>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <pre className="text-sm text-gray-900 dark:text-white whitespace-pre-wrap">
                    {selectedVariant.promptText}
                  </pre>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(selectedVariant.metrics).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <div className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                      {key}
                    </div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-white mt-1">
                      {typeof value === 'number' ? 
                        (key === 'accuracy' || key === 'coherence' || key === 'relevance' ? 
                          `${(value * 100).toFixed(1)}%` : 
                          key === 'latency' ? `${value.toFixed(0)}ms` :
                          key === 'cost' ? `${value.toFixed(4)}` :
                          value.toFixed(2)
                        ) : 
                        value
                      }
                    </div>
                  </div>
                ))}
              </div>

              {/* Metadata */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Configuration
                </h4>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Model:</span> {selectedVariant.metadata.model}
                    </div>
                    <div>
                      <span className="font-medium">Temperature:</span> {selectedVariant.metadata.temperature}
                    </div>
                    <div>
                      <span className="font-medium">Max Tokens:</span> {selectedVariant.metadata.maxTokens}
                    </div>
                    <div>
                      <span className="font-medium">Test Cases:</span> {selectedVariant.metadata.testCases}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDashboard;