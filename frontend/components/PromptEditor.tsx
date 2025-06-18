'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { 
  Save, 
  Download, 
  Upload, 
  Play, 
  GitBranch, 
  History, 
  Settings, 
  Copy, 
  Undo, 
  Redo,
  Bold,
  Italic,
  List,
  Hash,
  Eye,
  EyeOff,
  Zap,
  Check,
  AlertCircle,
  Loader2
} from 'lucide-react';

// Types and Interfaces
interface PromptVersion {
  id: string;
  content: string;
  timestamp: Date;
  branch: string;
  description?: string;
  optimizationScore?: number;
}

interface Variable {
  name: string;
  start: number;
  end: number;
}

interface AutoSaveStatus {
  status: 'saved' | 'saving' | 'error' | 'unsaved';
  lastSaved?: Date;
  error?: string;
}

interface OptimizationSettings {
  model: string;
  temperature: number;
  maxTokens: number;
  variants: number;
}

interface PromptEditorProps {
  /** Initial prompt content */
  initialContent?: string;
  /** Prompt ID for loading existing prompts */
  promptId?: string;
  /** Callback when prompt is saved */
  onSave?: (content: string, metadata: any) => void;
  /** Callback when optimization is triggered */
  onOptimize?: (content: string, settings: OptimizationSettings) => void;
  /** Read-only mode */
  readOnly?: boolean;
  /** Custom placeholder text */
  placeholder?: string;
  /** Maximum character limit */
  maxLength?: number;
}

/**
 * Custom hook for managing prompt editor state and operations
 */
const usePromptEditor = (initialContent: string = '', promptId?: string) => {
  const [content, setContent] = useState(initialContent);
  const [versions, setVersions] = useState<PromptVersion[]>([]);
  const [currentBranch, setCurrentBranch] = useState('main');
  const [autoSaveStatus, setAutoSaveStatus] = useState<AutoSaveStatus>({ status: 'saved' });
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [variables, setVariables] = useState<Variable[]>([]);
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);

  // Auto-save with debouncing
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout>();
  
  const debouncedAutoSave = useCallback(async (content: string) => {
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }
    
    autoSaveTimeoutRef.current = setTimeout(async () => {
      try {
        setAutoSaveStatus({ status: 'saving' });
        // Simulate API call - replace with actual API integration
        await new Promise(resolve => setTimeout(resolve, 500));
        setAutoSaveStatus({ status: 'saved', lastSaved: new Date() });
      } catch (error) {
        setAutoSaveStatus({ 
          status: 'error', 
          error: error instanceof Error ? error.message : 'Save failed' 
        });
      }
    }, 1000);
  }, []);

  // Extract variables from content
  const extractVariables = useCallback((text: string): Variable[] => {
    const regex = /\{\{(\w+)\}\}/g;
    const variables: Variable[] = [];
    let match;
    
    while ((match = regex.exec(text)) !== null) {
      variables.push({
        name: match[1],
        start: match.index,
        end: match.index + match[0].length
      });
    }
    
    return variables;
  }, []);

  // Update content with undo/redo support
  const updateContent = useCallback((newContent: string, addToHistory: boolean = true) => {
    if (addToHistory && content !== newContent) {
      setUndoStack(prev => [...prev, content]);
      setRedoStack([]);
    }
    
    setContent(newContent);
    setVariables(extractVariables(newContent));
    setAutoSaveStatus({ status: 'unsaved' });
    debouncedAutoSave(newContent);
  }, [content, extractVariables, debouncedAutoSave]);

  // Undo/Redo operations
  const undo = useCallback(() => {
    if (undoStack.length > 0) {
      const previous = undoStack[undoStack.length - 1];
      setRedoStack(prev => [content, ...prev]);
      setUndoStack(prev => prev.slice(0, -1));
      setContent(previous);
      setVariables(extractVariables(previous));
    }
  }, [undoStack, content, extractVariables]);

  const redo = useCallback(() => {
    if (redoStack.length > 0) {
      const next = redoStack[0];
      setUndoStack(prev => [...prev, content]);
      setRedoStack(prev => prev.slice(1));
      setContent(next);
      setVariables(extractVariables(next));
    }
  }, [redoStack, content, extractVariables]);

  return {
    content,
    updateContent,
    versions,
    setVersions,
    currentBranch,
    setCurrentBranch,
    autoSaveStatus,
    isOptimizing,
    setIsOptimizing,
    variables,
    undo,
    redo,
    canUndo: undoStack.length > 0,
    canRedo: redoStack.length > 0
  };
};

/**
 * Main PromptEditor component - A rich-text editor for prompt creation and editing
 * with version control, optimization features, and real-time collaboration support.
 */
export const PromptEditor: React.FC<PromptEditorProps> = ({
  initialContent = '',
  promptId,
  onSave,
  onOptimize,
  readOnly = false,
  placeholder = 'Start typing your prompt here...',
  maxLength = 10000
}) => {
  const {
    content,
    updateContent,
    versions,
    currentBranch,
    autoSaveStatus,
    isOptimizing,
    setIsOptimizing,
    variables,
    undo,
    redo,
    canUndo,
    canRedo
  } = usePromptEditor(initialContent, promptId);

  // UI State
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [optimizationSettings, setOptimizationSettings] = useState<OptimizationSettings>({
    model: 'gpt-4',
    temperature: 0.7,
    maxTokens: 1000,
    variants: 3
  });

  // Refs
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Computed values
  const tokenCount = useMemo(() => {
    // Rough estimation: 1 token ≈ 4 characters
    return Math.ceil(content.length / 4);
  }, [content]);

  const characterCount = content.length;
  const isNearLimit = maxLength && characterCount > maxLength * 0.9;

  // Event handlers
  const handleContentChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newContent = e.target.value;
    if (!maxLength || newContent.length <= maxLength) {
      updateContent(newContent);
    }
  }, [updateContent, maxLength]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'z':
          e.preventDefault();
          if (e.shiftKey) {
            redo();
          } else {
            undo();
          }
          break;
        case 's':
          e.preventDefault();
          handleSave();
          break;
        case 'Enter':
          if (e.shiftKey) {
            e.preventDefault();
            handleOptimize();
          }
          break;
      }
    }
  }, [undo, redo]);

  const handleSave = useCallback(async () => {
    try {
      if (onSave) {
        await onSave(content, { branch: currentBranch, variables });
      }
      // Add to version history
      const newVersion: PromptVersion = {
        id: Date.now().toString(),
        content,
        timestamp: new Date(),
        branch: currentBranch,
        description: `Saved at ${new Date().toLocaleTimeString()}`
      };
      // setVersions(prev => [newVersion, ...prev]);
    } catch (error) {
      console.error('Save failed:', error);
    }
  }, [content, currentBranch, variables, onSave]);

  const handleOptimize = useCallback(async () => {
    if (!content.trim() || isOptimizing) return;
    
    try {
      setIsOptimizing(true);
      if (onOptimize) {
        await onOptimize(content, optimizationSettings);
      }
    } catch (error) {
      console.error('Optimization failed:', error);
    } finally {
      setIsOptimizing(false);
    }
  }, [content, isOptimizing, optimizationSettings, onOptimize]);

  const handleImport = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        updateContent(content);
      };
      reader.readAsText(file);
    }
  }, [updateContent]);

  const handleExport = useCallback(() => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prompt-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [content]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
    } catch (error) {
      console.error('Copy failed:', error);
    }
  }, [content]);

  // Syntax highlighting for variables
  const highlightedContent = useMemo(() => {
    if (!content) return '';
    
    let highlighted = content;
    variables.forEach(variable => {
      const pattern = new RegExp(`\\{\\{${variable.name}\\}\\}`, 'g');
      highlighted = highlighted.replace(pattern, `<span class="variable">{{${variable.name}}}</span>`);
    });
    
    return highlighted;
  }, [content, variables]);

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <div className="flex items-center space-x-2">
          {/* Format buttons */}
          <div className="flex items-center space-x-1 border-r border-gray-300 dark:border-gray-600 pr-2">
            <button
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Bold (Ctrl+B)"
              disabled={readOnly}
            >
              <Bold className="w-4 h-4" />
            </button>
            <button
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Italic (Ctrl+I)"
              disabled={readOnly}
            >
              <Italic className="w-4 h-4" />
            </button>
            <button
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="List"
              disabled={readOnly}
            >
              <List className="w-4 h-4" />
            </button>
          </div>

          {/* Undo/Redo */}
          <div className="flex items-center space-x-1 border-r border-gray-300 dark:border-gray-600 pr-2">
            <button
              onClick={undo}
              disabled={!canUndo || readOnly}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Undo (Ctrl+Z)"
            >
              <Undo className="w-4 h-4" />
            </button>
            <button
              onClick={redo}
              disabled={!canRedo || readOnly}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Redo (Ctrl+Shift+Z)"
            >
              <Redo className="w-4 h-4" />
            </button>
          </div>

          {/* File operations */}
          <div className="flex items-center space-x-1">
            <button
              onClick={handleImport}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Import"
            >
              <Upload className="w-4 h-4" />
            </button>
            <button
              onClick={handleExport}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Export"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={handleCopy}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Copy"
            >
              <Copy className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* View toggles */}
          <button
            onClick={() => setShowPreview(!showPreview)}
            className={`p-2 rounded transition-colors ${
              showPreview
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title="Toggle Preview"
          >
            {showPreview ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>

          <button
            onClick={() => setShowHistory(!showHistory)}
            className={`p-2 rounded transition-colors ${
              showHistory
                ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title="Show History"
          >
            <History className="w-4 h-4" />
          </button>

          {/* Optimization */}
          <button
            onClick={handleOptimize}
            disabled={isOptimizing || !content.trim()}
            className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
            title="Optimize Prompt (Ctrl+Shift+Enter)"
          >
            {isOptimizing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Zap className="w-4 h-4" />
            )}
            <span className="font-medium">Optimize</span>
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Editor Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Main Editor */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={content}
              onChange={handleContentChange}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={readOnly}
              className="w-full h-full p-4 text-gray-900 dark:text-gray-100 bg-transparent resize-none border-none outline-none font-mono text-sm leading-relaxed"
              style={{ fontFamily: 'SF Mono, Monaco, Consolas, monospace' }}
            />
            
            {/* Variable highlighting overlay */}
            {variables.length > 0 && (
              <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                {variables.map((variable, index) => (
                  <span
                    key={index}
                    className="absolute bg-yellow-200 dark:bg-yellow-800 bg-opacity-30 rounded px-1"
                    style={{
                      // Calculate position based on content
                      top: `${Math.floor(variable.start / 80) * 1.5 + 1}rem`,
                      left: `${(variable.start % 80) * 0.6 + 1}rem`,
                    }}
                  >
                    {variable.name}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Side Panel */}
        {(showHistory || showSettings) && (
          <div className="w-80 border-l border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            {showHistory && (
              <div className="p-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Version History
                </h3>
                <div className="space-y-2">
                  {versions.length > 0 ? (
                    versions.map((version) => (
                      <div
                        key={version.id}
                        className="p-3 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors cursor-pointer"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {version.branch}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {version.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        {version.description && (
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            {version.description}
                          </p>
                        )}
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      No version history yet
                    </p>
                  )}
                </div>
              </div>
            )}

            {showSettings && (
              <div className="p-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Optimization Settings
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Model
                    </label>
                    <select
                      value={optimizationSettings.model}
                      onChange={(e) => setOptimizationSettings(prev => ({ ...prev, model: e.target.value }))}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    >
                      <option value="gpt-4">GPT-4</option>
                      <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                      <option value="claude-3">Claude 3</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Temperature: {optimizationSettings.temperature}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={optimizationSettings.temperature}
                      onChange={(e) => setOptimizationSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Tokens
                    </label>
                    <input
                      type="number"
                      value={optimizationSettings.maxTokens}
                      onChange={(e) => setOptimizationSettings(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Variants to Generate
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={optimizationSettings.variants}
                      onChange={(e) => setOptimizationSettings(prev => ({ ...prev, variants: parseInt(e.target.value) }))}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-sm">
        <div className="flex items-center space-x-4">
          {/* Save Status */}
          <div className="flex items-center space-x-2">
            {autoSaveStatus.status === 'saving' && (
              <>
                <Loader2 className="w-3 h-3 animate-spin text-blue-600" />
                <span className="text-blue-600">Saving...</span>
              </>
            )}
            {autoSaveStatus.status === 'saved' && (
              <>
                <Check className="w-3 h-3 text-green-600" />
                <span className="text-green-600">
                  Saved {autoSaveStatus.lastSaved?.toLocaleTimeString()}
                </span>
              </>
            )}
            {autoSaveStatus.status === 'error' && (
              <>
                <AlertCircle className="w-3 h-3 text-red-600" />
                <span className="text-red-600">Save failed</span>
              </>
            )}
            {autoSaveStatus.status === 'unsaved' && (
              <span className="text-gray-500 dark:text-gray-400">Unsaved changes</span>
            )}
          </div>

          {/* Branch Info */}
          <div className="flex items-center space-x-1 text-gray-500 dark:text-gray-400">
            <GitBranch className="w-3 h-3" />
            <span>{currentBranch}</span>
          </div>

          {/* Variables */}
          {variables.length > 0 && (
            <div className="flex items-center space-x-1 text-gray-500 dark:text-gray-400">
              <Hash className="w-3 h-3" />
              <span>{variables.length} variable{variables.length !== 1 ? 's' : ''}</span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-4 text-gray-500 dark:text-gray-400">
          {/* Token Count */}
          <span>~{tokenCount} tokens</span>
          
          {/* Character Count */}
          <span className={isNearLimit ? 'text-orange-600 dark:text-orange-400' : ''}>
            {characterCount}{maxLength ? `/${maxLength}` : ''} chars
          </span>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.md"
        onChange={handleFileImport}
        className="hidden"
      />
    </div>
  );
};

export default PromptEditor;