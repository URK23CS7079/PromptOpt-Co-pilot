import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  Grid3X3,
  List,
  GitCompare,
  Copy, 
  Download,
  Heart,
  HeartOff,
  Filter,
  Search,
  ArrowUpDown,
  Eye,
  BarChart3,
  Code,
  Clock,
  CheckCircle2,
  XCircle,
  TrendingUp,
  Settings,
  ChevronDown,
  ChevronUp,
  Maximize2,
  Minimize2
} from 'lucide-react';

// Types and Interfaces
export interface VariantMetrics {
  score: number;
  latency: number;
  successRate: number;
  accuracy?: number;
  cost?: number;
  tokens?: number;
}

export interface PromptVariant {
  id: string;
  content: string;
  title: string;
  description?: string;
  createdAt: Date;
  updatedAt: Date;
  parentId?: string;
  optimizationType: 'APE' | 'DSPy' | 'Manual';
  metrics: VariantMetrics;
  variables: string[];
  tags: string[];
  isFavorite: boolean;
  isBaseline?: boolean;
}

export interface ComparisonResult {
  variantId: string;
  differences: DiffLine[];
  similarity: number;
}

export interface DiffLine {
  type: 'added' | 'removed' | 'unchanged' | 'modified';
  content: string;
  lineNumber: number;
}

export interface VariantViewerProps {
  variants: PromptVariant[];
  loading?: boolean;
  onVariantSelect?: (variant: PromptVariant) => void;
  onVariantCompare?: (variants: PromptVariant[]) => void;
  onVariantFavorite?: (variantId: string, isFavorite: boolean) => void;
  onVariantExport?: (variant: PromptVariant) => void;
  className?: string;
}

// Utility functions
const highlightVariables = (text: string, variables: string[]): string => {
  let highlightedText = text;
  variables.forEach(variable => {
    const regex = new RegExp(`\\{${variable}\\}`, 'gi');
    highlightedText = highlightedText.replace(
      regex, 
      `<span class="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-1 rounded">{${variable}}</span>`
    );
  });
  return highlightedText;
};

const calculateDiff = (original: string, modified: string): DiffLine[] => {
  const originalLines = original.split('\n');
  const modifiedLines = modified.split('\n');
  const diff: DiffLine[] = [];
  
  // Simple diff algorithm (in production, use a proper diff library)
  const maxLength = Math.max(originalLines.length, modifiedLines.length);
  
  for (let i = 0; i < maxLength; i++) {
    const originalLine = originalLines[i] || '';
    const modifiedLine = modifiedLines[i] || '';
    
    if (originalLine === modifiedLine) {
      diff.push({
        type: 'unchanged',
        content: originalLine,
        lineNumber: i + 1
      });
    } else if (!originalLine) {
      diff.push({
        type: 'added',
        content: modifiedLine,
        lineNumber: i + 1
      });
    } else if (!modifiedLine) {
      diff.push({
        type: 'removed',
        content: originalLine,
        lineNumber: i + 1
      });
    } else {
      diff.push({
        type: 'modified',
        content: modifiedLine,
        lineNumber: i + 1
      });
    }
  }
  
  return diff;
};

// Sub-components

/**
 * VariantMetrics Component
 * Displays performance metrics for a variant
 */
interface VariantMetricsProps {
  metrics: VariantMetrics;
  compact?: boolean;
  showTrends?: boolean;
}

const VariantMetrics: React.FC<VariantMetricsProps> = ({ 
  metrics, 
  compact = false,
  showTrends = false 
}) => {
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400';
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const formatLatency = (latency: number) => {
    if (latency < 1000) return `${latency}ms`;
    return `${(latency / 1000).toFixed(1)}s`;
  };

  if (compact) {
    return (
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-1">
          <BarChart3 className="w-3 h-3" />
          <span className={getScoreColor(metrics.score)}>
            {(metrics.score * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          <span>{formatLatency(metrics.latency)}</span>
        </div>
        <div className="flex items-center gap-1">
          <CheckCircle2 className="w-3 h-3" />
          <span>{(metrics.successRate * 100).toFixed(1)}%</span>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div className="text-center">
        <div className="flex items-center justify-center gap-1 mb-1">
          <BarChart3 className="w-4 h-4" />
          <span className="text-sm font-medium">Score</span>
          {showTrends && <TrendingUp className="w-3 h-3 text-green-500" />}
        </div>
        <div className={`text-lg font-bold ${getScoreColor(metrics.score)}`}>
          {(metrics.score * 100).toFixed(1)}%
        </div>
      </div>
      
      <div className="text-center">
        <div className="flex items-center justify-center gap-1 mb-1">
          <Clock className="w-4 h-4" />
          <span className="text-sm font-medium">Latency</span>
        </div>
        <div className="text-lg font-bold">
          {formatLatency(metrics.latency)}
        </div>
      </div>
      
      <div className="text-center">
        <div className="flex items-center justify-center gap-1 mb-1">
          <CheckCircle2 className="w-4 h-4" />
          <span className="text-sm font-medium">Success Rate</span>
        </div>
        <div className="text-lg font-bold">
          {(metrics.successRate * 100).toFixed(1)}%
        </div>
      </div>
      
      {metrics.accuracy && (
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 mb-1">
            <Eye className="w-4 h-4" />
            <span className="text-sm font-medium">Accuracy</span>
          </div>
          <div className="text-lg font-bold">
            {(metrics.accuracy * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * VariantActions Component
 * Provides action buttons for variant operations
 */
interface VariantActionsProps {
  variant: PromptVariant;
  onFavorite?: (isFavorite: boolean) => void;
  onCopy?: () => void;
  onExport?: () => void;
  onView?: () => void;
  compact?: boolean;
}

const VariantActions: React.FC<VariantActionsProps> = ({
  variant,
  onFavorite,
  onCopy,
  onExport,
  onView,
  compact = false
}) => {
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(variant.content);
    onCopy?.();
  }, [variant.content, onCopy]);

  const actionClasses = compact 
    ? "p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
    : "p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700";

  return (
    <div className={`flex items-center gap-1 ${compact ? '' : 'gap-2'}`}>
      <button
        onClick={() => onFavorite?.(!variant.isFavorite)}
        className={`${actionClasses} transition-colors`}
        title={variant.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
      >
        {variant.isFavorite ? (
          <Heart className={`${compact ? 'w-3 h-3' : 'w-4 h-4'} text-red-500 fill-current`} />
        ) : (
          <HeartOff className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
        )}
      </button>
      
      <button
        onClick={handleCopy}
        className={`${actionClasses} transition-colors`}
        title="Copy to clipboard"
      >
        <Copy className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
      </button>
      
      <button
        onClick={onExport}
        className={`${actionClasses} transition-colors`}
        title="Export variant"
      >
        <Download className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
      </button>
      
      {onView && (
        <button
          onClick={onView}
          className={`${actionClasses} transition-colors`}
          title="View details"
        >
          <Eye className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
        </button>
      )}
    </div>
  );
};

/**
 * VariantCard Component
 * Displays individual variant in card format
 */
interface VariantCardProps {
  variant: PromptVariant;
  selected?: boolean;
  onSelect?: () => void;
  onFavorite?: (isFavorite: boolean) => void;
  onExport?: () => void;
  showMetrics?: boolean;
  compact?: boolean;
}

const VariantCard: React.FC<VariantCardProps> = ({
  variant,
  selected = false,
  onSelect,
  onFavorite,
  onExport,
  showMetrics = true,
  compact = false
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const previewText = useMemo(() => {
    return variant.content.length > 200 
      ? variant.content.substring(0, 200) + '...'
      : variant.content;
  }, [variant.content]);

  const cardClasses = `
    border rounded-lg p-4 transition-all cursor-pointer
    ${selected 
      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
    }
    ${compact ? 'p-3' : 'p-4'}
  `;

  return (
    <div className={cardClasses} onClick={onSelect}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className={`font-semibold ${compact ? 'text-sm' : 'text-base'}`}>
            {variant.title}
          </h3>
          {variant.description && (
            <p className={`text-gray-600 dark:text-gray-400 ${compact ? 'text-xs' : 'text-sm'} mt-1`}>
              {variant.description}
            </p>
          )}
          <div className="flex items-center gap-2 mt-2">
            <span className={`px-2 py-1 rounded text-xs font-medium
              ${variant.optimizationType === 'APE' 
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : variant.optimizationType === 'DSPy'
                ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
              }`}>
              {variant.optimizationType}
            </span>
            {variant.isBaseline && (
              <span className="px-2 py-1 rounded text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                Baseline
              </span>
            )}
          </div>
        </div>
        
        <VariantActions
          variant={variant}
          onFavorite={onFavorite}
          onExport={onExport}
          compact={compact}
        />
      </div>

      <div className="mb-3">
        <div className="flex items-center justify-between mb-2">
          <span className={`text-gray-600 dark:text-gray-400 ${compact ? 'text-xs' : 'text-sm'}`}>
            Prompt Content
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
        
        <div className={`bg-gray-50 dark:bg-gray-800 rounded p-3 ${compact ? 'text-xs' : 'text-sm'} font-mono`}>
          <div 
            dangerouslySetInnerHTML={{
              __html: highlightVariables(
                isExpanded ? variant.content : previewText,
                variant.variables
              )
            }}
          />
        </div>
        
        {variant.variables.length > 0 && (
          <div className="mt-2">
            <span className={`text-gray-600 dark:text-gray-400 ${compact ? 'text-xs' : 'text-sm'}`}>
              Variables: 
            </span>
            <div className="flex flex-wrap gap-1 mt-1">
              {variant.variables.map((variable, index) => (
                <span 
                  key={index}
                  className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs"
                >
                  {variable}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {showMetrics && (
        <VariantMetrics metrics={variant.metrics} compact={compact} />
      )}
    </div>
  );
};

/**
 * VariantDiff Component
 * Shows differences between variants
 */
interface VariantDiffProps {
  originalVariant: PromptVariant;
  modifiedVariant: PromptVariant;
  showLineNumbers?: boolean;
}

const VariantDiff: React.FC<VariantDiffProps> = ({
  originalVariant,
  modifiedVariant,
  showLineNumbers = true
}) => {
  const diff = useMemo(() => 
    calculateDiff(originalVariant.content, modifiedVariant.content),
    [originalVariant.content, modifiedVariant.content]
  );

  const getDiffLineClass = (type: DiffLine['type']) => {
    switch (type) {
      case 'added':
        return 'bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500';
      case 'removed':
        return 'bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500';
      case 'modified':
        return 'bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500';
      default:
        return 'bg-gray-50 dark:bg-gray-800';
    }
  };

  const getDiffIcon = (type: DiffLine['type']) => {
    switch (type) {
      case 'added':
        return <span className="text-green-600 dark:text-green-400">+</span>;
      case 'removed':
        return <span className="text-red-600 dark:text-red-400">-</span>;
      case 'modified':
        return <span className="text-yellow-600 dark:text-yellow-400">~</span>;
      default:
        return <span className="text-gray-400"> </span>;
    }
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="bg-gray-100 dark:bg-gray-800 px-4 py-2 border-b">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">
            Diff: {originalVariant.title} → {modifiedVariant.title}
          </h3>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span>Added</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>Removed</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-yellow-500 rounded"></div>
              <span>Modified</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {diff.map((line, index) => (
          <div 
            key={index}
            className={`flex items-start gap-2 p-2 ${getDiffLineClass(line.type)}`}
          >
            {showLineNumbers && (
              <span className="text-gray-400 text-sm font-mono w-8 text-right">
                {line.lineNumber}
              </span>
            )}
            <span className="w-4 font-mono text-sm">
              {getDiffIcon(line.type)}
            </span>
            <span className="flex-1 font-mono text-sm whitespace-pre-wrap">
              {line.content}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * VariantComparison Component
 * Side-by-side comparison of variants
 */
interface VariantComparisonProps {
  variants: PromptVariant[];
  onRemoveVariant?: (variantId: string) => void;
}

const VariantComparison: React.FC<VariantComparisonProps> = ({
  variants,
  onRemoveVariant
}) => {
  const [selectedTab, setSelectedTab] = useState<'content' | 'metrics' | 'diff'>('content');

  if (variants.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        Select variants to compare
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">
          Comparing {variants.length} Variants
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSelectedTab('content')}
            className={`px-3 py-1 rounded text-sm ${
              selectedTab === 'content'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            Content
          </button>
          <button
            onClick={() => setSelectedTab('metrics')}
            className={`px-3 py-1 rounded text-sm ${
              selectedTab === 'metrics'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            Metrics
          </button>
          {variants.length === 2 && (
            <button
              onClick={() => setSelectedTab('diff')}
              className={`px-3 py-1 rounded text-sm ${
                selectedTab === 'diff'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              Diff
            </button>
          )}
        </div>
      </div>

      {selectedTab === 'content' && (
        <div className={`grid gap-4 ${variants.length === 2 ? 'grid-cols-2' : 'grid-cols-1'}`}>
          {variants.map((variant) => (
            <div key={variant.id} className="border rounded-lg">
              <div className="bg-gray-50 dark:bg-gray-800 px-4 py-2 border-b flex items-center justify-between">
                <h4 className="font-semibold">{variant.title}</h4>
                <button
                  onClick={() => onRemoveVariant?.(variant.id)}
                  className="text-red-600 hover:text-red-800"
                >
                  <XCircle className="w-4 h-4" />
                </button>
              </div>
              <div className="p-4">
                <div className="bg-gray-50 dark:bg-gray-800 rounded p-3 text-sm font-mono">
                  <div 
                    dangerouslySetInnerHTML={{
                      __html: highlightVariables(variant.content, variant.variables)
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedTab === 'metrics' && (
        <div className="space-y-4">
          {variants.map((variant) => (
            <div key={variant.id} className="border rounded-lg p-4">
              <h4 className="font-semibold mb-3">{variant.title}</h4>
              <VariantMetrics metrics={variant.metrics} showTrends />
            </div>
          ))}
        </div>
      )}

      {selectedTab === 'diff' && variants.length === 2 && (
        <VariantDiff
          originalVariant={variants[0]}
          modifiedVariant={variants[1]}
        />
      )}
    </div>
  );
};

/**
 * Main VariantViewer Component
 */
export const VariantViewer: React.FC<VariantViewerProps> = ({
  variants,
  loading = false,
  onVariantSelect,
  onVariantCompare,
  onVariantFavorite,
  onVariantExport,
  className = ''
}) => {
  // State management
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [selectedVariants, setSelectedVariants] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'score' | 'latency' | 'successRate' | 'createdAt'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterBy, setFilterBy] = useState<'all' | 'favorites' | 'APE' | 'DSPy' | 'Manual'>('all');
  const [showFilters, setShowFilters] = useState(false);

  // Filtered and sorted variants
  const filteredVariants = useMemo(() => {
    let filtered = variants.filter(variant => {
      // Search filter
      const matchesSearch = variant.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           variant.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           (variant.description?.toLowerCase().includes(searchTerm.toLowerCase()) ?? false);
      
      // Type filter
      const matchesFilter = filterBy === 'all' || 
                           (filterBy === 'favorites' && variant.isFavorite) ||
                           variant.optimizationType === filterBy;
      
      return matchesSearch && matchesFilter;
    });

    // Sort variants
    filtered.sort((a, b) => {
      let aValue: number | Date;
      let bValue: number | Date;
      
      switch (sortBy) {
        case 'score':
          aValue = a.metrics.score;
          bValue = b.metrics.score;
          break;
        case 'latency':
          aValue = a.metrics.latency;
          bValue = b.metrics.latency;
          break;
        case 'successRate':
          aValue = a.metrics.successRate;
          bValue = b.metrics.successRate;
          break;
        case 'createdAt':
          aValue = a.createdAt;
          bValue = b.createdAt;
          break;
        default:
          return 0;
      }
      
      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });

    return filtered;
  }, [variants, searchTerm, sortBy, sortOrder, filterBy]);

  // Event handlers
  const handleVariantSelect = useCallback((variant: PromptVariant) => {
    if (isCompareMode) {
      setSelectedVariants(prev => {
        if (prev.includes(variant.id)) {
          return prev.filter(id => id !== variant.id);
        } else if (prev.length < 3) {
          return [...prev, variant.id];
        }
        return prev;
      });
    } else {
      onVariantSelect?.(variant);
    }
  }, [isCompareMode, onVariantSelect]);

  const handleToggleCompareMode = useCallback(() => {
    setIsCompareMode(prev => !prev);
    if (!isCompareMode) {
      setSelectedVariants([]);
    }
  }, [isCompareMode]);

  const handleCompareVariants = useCallback(() => {
    const selectedVariantObjects = variants.filter(v => selectedVariants.includes(v.id));
    onVariantCompare?.(selectedVariantObjects);
  }, [variants, selectedVariants, onVariantCompare]);

  const handleFavoriteVariant = useCallback((variantId: string, isFavorite: boolean) => {
    onVariantFavorite?.(variantId, isFavorite);
  }, [onVariantFavorite]);

  const handleExportVariant = useCallback((variant: PromptVariant) => {
    onVariantExport?.(variant);
  }, [onVariantExport]);

  if (loading) {
    return (
      <div className={`flex items-center justify-center py-12 ${className}`}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header Controls */}
      <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold">
            Prompt Variants ({filteredVariants.length})
          </h2>
          
          {isCompareMode && selectedVariants.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {selectedVariants.length} selected
              </span>
              <button
                onClick={handleCompareVariants}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors"
              >
                Compare
              </button>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search variants..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          {/* Filter Toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-2 rounded-lg transition-colors ${
              showFilters 
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            <Filter className="w-4 h-4" />
          </button>
          
          {/* Compare Mode Toggle */}
          <button
            onClick={handleToggleCompareMode}
            className={`p-2 rounded-lg transition-colors ${
              isCompareMode 
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title="Toggle compare mode"
          >
            <GitCompare className="w-4 h-4" />
          </button>
          
          {/* View Mode Toggle */}
          <div className="flex items-center border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 transition-colors ${
                viewMode === 'grid'
                  ? 'bg-blue-600 text-white'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
              title="Grid view"
            >
              <Grid3X3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 transition-colors ${
                viewMode === 'list'
                  ? 'bg-blue-600 text-white'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
              title="List view"
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Sort By */}
            <div>
              <label className="block text-sm font-medium mb-2">Sort By</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
              >
                <option value="score">Score</option>
                <option value="latency">Latency</option>
                <option value="successRate">Success Rate</option>
                <option value="createdAt">Created Date</option>
              </select>
            </div>
            
            {/* Sort Order */}
            <div>
              <label className="block text-sm font-medium mb-2">Order</label>
              <select
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value as typeof sortOrder)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
              >
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </div>
            
            {/* Filter By */}
            <div>
              <label className="block text-sm font-medium mb-2">Filter By</label>
              <select
                value={filterBy}
                onChange={(e) => setFilterBy(e.target.value as typeof filterBy)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
              >
                <option value="all">All Variants</option>
                <option value="favorites">Favorites</option>
                <option value="APE">APE Optimized</option>
                <option value="DSPy">DSPy Optimized</option>
                <option value="Manual">Manual</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Compare Mode Panel */}
      {isCompareMode && selectedVariants.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <VariantComparison
            variants={variants.filter(v => selectedVariants.includes(v.id))}
            onRemoveVariant={(variantId) => 
              setSelectedVariants(prev => prev.filter(id => id !== variantId))
            }
          />
        </div>
      )}

      {/* Variants Display */}
      {filteredVariants.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <Code className="w-12 h-12 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No variants found
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            {searchTerm || filterBy !== 'all' 
              ? 'Try adjusting your search or filters'
              : 'No prompt variants available yet'
            }
          </p>
        </div>
      ) : (
        <div className={
          viewMode === 'grid'
            ? 'grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4'
            : 'space-y-4'
        }>
          {filteredVariants.map((variant) => (
            <VariantCard
              key={variant.id}
              variant={variant}
              selected={isCompareMode && selectedVariants.includes(variant.id)}
              onSelect={() => handleVariantSelect(variant)}
              onFavorite={(isFavorite) => handleFavoriteVariant(variant.id, isFavorite)}
              onExport={() => handleExportVariant(variant)}
              showMetrics={true}
              compact={viewMode === 'list'}
            />
          ))}
        </div>
      )}

      {/* Load More / Pagination could go here */}
      {filteredVariants.length > 0 && (
        <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Showing {filteredVariants.length} of {variants.length} variants
          </div>
          
          {/* Export All Button */}
          <button
            onClick={() => {
              // Export all filtered variants
              filteredVariants.forEach(variant => handleExportVariant(variant));
            }}
            className="flex items-center gap-2 px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export All
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * Default export
 */
export default VariantViewer;

/**
 * Unit Tests
 * These would typically be in a separate .test.tsx file
 */

// Mock data for testing
export const mockVariants: PromptVariant[] = [
  {
    id: '1',
    title: 'Original Prompt',
    description: 'The baseline prompt for comparison',
    content: 'Please analyze the following text: {text}\n\nProvide a summary in {format} format.',
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-01'),
    optimizationType: 'Manual',
    metrics: {
      score: 0.75,
      latency: 1200,
      successRate: 0.85,
      accuracy: 0.78
    },
    variables: ['text', 'format'],
    tags: ['baseline', 'summary'],
    isFavorite: false,
    isBaseline: true
  },
  {
    id: '2',
    title: 'APE Optimized v1',
    description: 'Optimized using Automatic Prompt Engineering',
    content: 'Analyze the provided text comprehensively: {text}\n\nGenerate a detailed summary using {format} structure with key insights highlighted.',
    createdAt: new Date('2024-01-02'),
    updatedAt: new Date('2024-01-02'),
    optimizationType: 'APE',
    metrics: {
      score: 0.89,
      latency: 1350,
      successRate: 0.92,
      accuracy: 0.87
    },
    variables: ['text', 'format'],
    tags: ['optimized', 'ape', 'summary'],
    isFavorite: true,
    parentId: '1'
  },
  {
    id: '3',
    title: 'DSPy Enhanced',
    description: 'Enhanced using DSPy optimization framework',
    content: 'Task: Comprehensive text analysis\nInput: {text}\nOutput format: {format}\n\nInstructions:\n1. Read and understand the text\n2. Extract key themes and concepts\n3. Structure findings in requested format\n4. Highlight critical insights',
    createdAt: new Date('2024-01-03'),
    updatedAt: new Date('2024-01-03'),
    optimizationType: 'DSPy',
    metrics: {
      score: 0.94,
      latency: 1450,
      successRate: 0.96,
      accuracy: 0.91
    },
    variables: ['text', 'format'],
    tags: ['optimized', 'dspy', 'summary', 'structured'],
    isFavorite: true,
    parentId: '1'
  }
];

/**
 * Test utility functions
 */
export const testUtils = {
  /**
   * Test variant filtering functionality
   */
  testVariantFiltering: () => {
    const searchResults = mockVariants.filter(v => 
      v.title.toLowerCase().includes('ape')
    );
    console.assert(searchResults.length === 1, 'Search filtering failed');
    
    const favoriteResults = mockVariants.filter(v => v.isFavorite);
    console.assert(favoriteResults.length === 2, 'Favorite filtering failed');
    
    console.log('✅ Variant filtering tests passed');
  },

  /**
   * Test variant sorting functionality
   */
  testVariantSorting: () => {
    const sortedByScore = [...mockVariants].sort((a, b) => b.metrics.score - a.metrics.score);
    console.assert(sortedByScore[0].id === '3', 'Score sorting failed');
    
    const sortedByLatency = [...mockVariants].sort((a, b) => a.metrics.latency - b.metrics.latency);
    console.assert(sortedByLatency[0].id === '1', 'Latency sorting failed');
    
    console.log('✅ Variant sorting tests passed');
  },

  /**
   * Test diff calculation
   */
  testDiffCalculation: () => {
    const diff = calculateDiff(mockVariants[0].content, mockVariants[1].content);
    console.assert(diff.length > 0, 'Diff calculation failed');
    console.assert(diff.some(line => line.type === 'modified'), 'No modifications detected');
    
    console.log('✅ Diff calculation tests passed');
  },

  /**
   * Test variable highlighting
   */
  testVariableHighlighting: () => {
    const highlighted = highlightVariables(
      'Test {variable1} and {variable2}',
      ['variable1', 'variable2']
    );
    console.assert(highlighted.includes('bg-blue-100'), 'Variable highlighting failed');
    
    console.log('✅ Variable highlighting tests passed');
  },

  /**
   * Run all tests
   */
  runAllTests: () => {
    console.log('🧪 Running VariantViewer tests...');
    testUtils.testVariantFiltering();
    testUtils.testVariantSorting();
    testUtils.testDiffCalculation();
    testUtils.testVariableHighlighting();
    console.log('✅ All VariantViewer tests passed!');
  }
};

// Export test utilities for use in testing environment
if (typeof window !== 'undefined' && (window as any).__TESTING__) {
  (window as any).VariantViewerTests = testUtils;
}