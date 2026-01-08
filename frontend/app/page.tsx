"use client";
import React, { useState, useEffect } from 'react';
import { Upload, Shield, Zap, TrendingUp, AlertCircle, CheckCircle, Lock, Database, Cpu, BarChart3, FileText, Download, ChevronRight, Brain, Server } from 'lucide-react';

const QuantumGuardAI = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [showDemo, setShowDemo] = useState(false);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:5001";


  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0];
  if (!file) return;

  setUploadedFile(file);
  setAnalyzing(true);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error("Analysis failed");
    }

    const data = await res.json();
    setResults(normalizeResults(data));

  } catch (err) {
    console.error(err);
    alert("Failed to analyze file");
  } finally {
    setAnalyzing(false);
  }
};

  // // Simulate file upload and analysis
  // const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const file = event.target.files[0];
  //   if (file) {
  //     setUploadedFile(file);
  //     setAnalyzing(true);
      
  //     // Simulate analysis
  //     setTimeout(() => {
  //       setResults({
  //         totalRecords: 1247,
  //         riskDistribution: {
  //           critical: 156,
  //           high: 423,
  //           medium: 389,
  //           low: 279
  //         },
  //         recommendations: {
  //           postquantum: 579,
  //           hybrid: 389,
  //           classical: 279
  //         },
  //         savings: {
  //           storageReduction: 35,
  //           costSaving: 68,
  //           securityRetained: 91
  //         },
  //         predictions: [
  //           { device: 'Health Monitor #A2301', risk: 'critical', current: 'RSA-2048', recommended: 'Kyber768', retention: '20 years' },
  //           { device: 'Smart Home #B7721', risk: 'low', current: 'ECC-256', recommended: 'Keep Classical', retention: '2 years' },
  //           { device: 'Industrial Sensor #C9912', risk: 'high', current: 'RSA-3072', recommended: 'Kyber768', retention: '12 years' },
  //           { device: 'Location Tracker #D4432', risk: 'medium', current: 'ECC-384', recommended: 'Hybrid (ECDH+Kyber)', retention: '5 years' }
  //         ]
  //       });
  //       setAnalyzing(false);
  //     }, 3000);
  //   }
  // };

  // Demo data simulation
  const runDemo = () => {
    setShowDemo(true);
    setAnalyzing(true);
    setTimeout(() => {
      setResults({
        totalRecords: 2500,
        riskDistribution: {
          critical: 325,
          high: 875,
          medium: 750,
          low: 550
        },
        recommendations: {
          postquantum: 1200,
          hybrid: 750,
          classical: 550
        },
        savings: {
          storageReduction: 35,
          costSaving: 68,
          securityRetained: 91
        },
        predictions: [
          { device: 'Health Monitor #A2301', risk: 'critical', current: 'RSA-2048', recommended: 'Kyber768', retention: '20 years' },
          { device: 'Smart Home #B7721', risk: 'low', current: 'ECC-256', recommended: 'Keep Classical', retention: '2 years' },
          { device: 'Industrial Sensor #C9912', risk: 'high', current: 'RSA-3072', recommended: 'Kyber768', retention: '12 years' },
          { device: 'Location Tracker #D4432', risk: 'medium', current: 'ECC-384', recommended: 'Hybrid (ECDH+Kyber)', retention: '5 years' },
          { device: 'Environmental Sensor #E1156', risk: 'medium', current: 'RSA-2048', recommended: 'Hybrid (ECDH+Kyber)', retention: '7 years' }
        ]
      });
      setAnalyzing(false);
    }, 3000);
  };

type RiskLevel = "critical" | "high" | "medium" | "low";

const getRiskColor = (risk: RiskLevel) => {
  const colors: Record<RiskLevel, string> = {
    critical: 'text-red-600 bg-red-50 border-red-200',
    high: 'text-orange-600 bg-orange-50 border-orange-200',
    medium: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    low: 'text-green-600 bg-green-50 border-green-200'
  };

  return colors[risk];
};

type GradientColor = "blue" | "purple" | "green" | "orange" | "pink" | "cyan";
const gradientMap: Record<GradientColor, string> = {
  blue: "from-blue-500 to-cyan-500",
  purple: "from-purple-500 to-pink-500",
  green: "from-green-500 to-emerald-500",
  orange: "from-orange-500 to-red-500",
  pink: "from-pink-500 to-rose-500",
  cyan: "from-cyan-500 to-blue-500",
};

const featureColorMap: Record<GradientColor, string> = {
  blue: "from-blue-500 to-blue-600",
  purple: "from-purple-500 to-purple-600",
  green: "from-green-500 to-green-600",
  orange: "from-orange-500 to-orange-600",
  pink: "from-pink-500 to-pink-600",
  cyan: "from-cyan-500 to-cyan-600",
};

// const featureColorMap: Record<GradientColor, string> = {
//   blue: "from-blue-500 to-blue-600",
//   purple: "from-purple-500 to-purple-600",
//   green: "from-green-500 to-green-600",
//   orange: "from-orange-500 to-orange-600",
//   pink: "from-pink-500 to-pink-600",
//   cyan: "from-cyan-500 to-cyan-600",
// };

  
  // const gradientMap = {
  //   blue: "from-blue-500 to-blue-600",
  //   purple: "from-purple-500 to-purple-600",
  //   green: "from-green-500 to-green-600",
  //   orange: "from-orange-500 to-orange-600",
  //   pink: "from-pink-500 to-pink-600",
  //   cyan: "from-cyan-500 to-cyan-600"
  // };


  const downloadMigrationPlan = async () => {
  const res = await fetch(`${API_BASE}/api/export/migration-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results)
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "migration_plan.csv";
  a.click();

  window.URL.revokeObjectURL(url);
};

const downloadReport = async () => {
  const res = await fetch(`${API_BASE}/api/export/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results)
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "quantumguard_report.json";
  a.click();

  window.URL.revokeObjectURL(url);
};

const normalizeResults = (data: any) => ({
  totalRecords: data.total_records ?? data.totalRecords,
  riskDistribution: data.summary?.risk_distribution ?? data.riskDistribution,
  recommendations: data.recommendations,
  savings: {
    costSaving: data.summary?.savings?.cost_saving_percent ?? data.savings?.costSaving,
    securityRetained: data.summary?.savings?.security_retained_percent ?? data.savings?.securityRetained,
    storageReduction: data.summary?.savings?.storage_reduction_percent ?? data.savings?.storageReduction
  },
  predictions: data.predictions
});





  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-lg border-b border-blue-800/30 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-2 rounded-lg">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  QuantumGuard AI
                </h1>
                <p className="text-xs text-slate-400">Intelligent Post-Quantum Migration</p>
              </div>
            </div>
            
            <nav className="flex space-x-1">
              {['home', 'demo', 'features', 'pricing'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === tab
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Home Tab */}
      {activeTab === 'home' && (
        <div className="max-w-7xl mx-auto px-6 py-16">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <div className="inline-block mb-4">
              <span className="bg-blue-500/20 text-blue-300 px-4 py-2 rounded-full text-sm font-semibold border border-blue-500/30">
                🚀 Powered by Machine Learning + Post-Quantum Cryptography
              </span>
            </div>
            <h2 className="text-5xl font-bold text-white mb-6 leading-tight">
              Protect Your IoT Data from<br />
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Future Quantum Threats
              </span>
            </h2>
            <p className="text-xl text-slate-300 mb-8 max-w-3xl mx-auto">
              AI-powered risk assessment that intelligently migrates your encrypted IoT data to post-quantum cryptography—reducing costs by 35% while maintaining 91% security.
            </p>
            
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setActiveTab('demo')}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-semibold text-lg flex items-center space-x-2 transition-all transform hover:scale-105 shadow-xl"
              >
                <Zap className="w-5 h-5" />
                <span>Try Live Demo</span>
              </button>
              <button
                onClick={() => setActiveTab('features')}
                className="bg-slate-800 hover:bg-slate-700 text-white px-8 py-4 rounded-xl font-semibold text-lg flex items-center space-x-2 transition-all border border-slate-700"
              >
                <span>Learn More</span>
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-4 gap-6 mb-16">
            {[
              { label: 'Cost Reduction', value: '35%', icon: TrendingUp, color: 'from-green-500 to-emerald-600' },
              { label: 'Security Retained', value: '91%', icon: Shield, color: 'from-blue-500 to-cyan-600' },
              { label: 'ML Accuracy', value: '85.8%', icon: Brain, color: 'from-purple-500 to-pink-600' },
              { label: 'Avg Analysis', value: '<2 min', icon: Zap, color: 'from-orange-500 to-red-600' }
            ].map((stat, i) => (
              <div key={i} className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 transition-all">
                <div className={`bg-gradient-to-br ${stat.color} w-12 h-12 rounded-lg flex items-center justify-center mb-4`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                <div className="text-sm text-slate-400">{stat.label}</div>
              </div>
            ))}
          </div>

          {/* Problem Statement */}
          <div className="bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-500/30 rounded-2xl p-8 mb-12">
            <div className="flex items-start space-x-4">
              <div className="bg-red-500/20 p-3 rounded-lg">
                <AlertCircle className="w-8 h-8 text-red-400" />
              </div>
              <div className="flex-1">
                <h3 className="text-2xl font-bold text-white mb-3">The Quantum Threat is Real</h3>
                <p className="text-slate-300 mb-4">
                  Encrypted IoT data stored today can be harvested and decrypted by future quantum computers. Healthcare records, financial data, and industrial logs with 10-20 year retention periods are vulnerable to "harvest now, decrypt later" attacks.
                </p>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-slate-900/50 rounded-lg p-4">
                    <div className="text-red-400 font-bold text-lg mb-1">RSA-2048</div>
                    <div className="text-sm text-slate-400">Breakable in ~5 years</div>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-4">
                    <div className="text-orange-400 font-bold text-lg mb-1">ECC-256</div>
                    <div className="text-sm text-slate-400">Breakable in ~6 years</div>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-4">
                    <div className="text-yellow-400 font-bold text-lg mb-1">Your Data</div>
                    <div className="text-sm text-slate-400">At risk right now</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* How It Works */}
          <div className="mb-16">
            <h3 className="text-3xl font-bold text-white text-center mb-12">How QuantumGuard AI Works</h3>
            <div className="grid grid-cols-4 gap-6">
              {[
              { step: 1, title: 'Upload Data', desc: 'Upload your IoT metadata (CSV/JSON)', icon: Upload, color: 'blue' },
                { step: 2, title: 'AI Analysis', desc: 'ML model predicts quantum risk', icon: Brain, color: 'purple' },
                { step: 3, title: 'Smart Migration', desc: 'Selective PQ encryption applied', icon: Lock, color: 'green' },
                { step: 4, title: 'Get Results', desc: 'Download migration plan + savings', icon: Download, color: 'orange' }
              ].map((item, i) => (
                <div key={i} className="relative">
                  <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 transition-all h-full">
                    <div className={`bg-gradient-to-br ${gradientMap[item.color as GradientColor]} w-12 h-12 rounded-full flex items-center justify-center mb-4 text-white font-bold text-lg`}>
                      {item.step}
                    </div>
                    <item.icon className="w-8 h-8 text-blue-400 mb-3" />
                    <h4 className="text-lg font-bold text-white mb-2">{item.title}</h4>
                    <p className="text-sm text-slate-400">{item.desc}</p>
                  </div>
                  {i < 3 && (
                    <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                      <ChevronRight className="w-6 h-6 text-slate-600" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Demo Tab */}
      {activeTab === 'demo' && (
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="text-center mb-8">
            <h2 className="text-4xl font-bold text-white mb-4">Live Demo</h2>
            <p className="text-xl text-slate-300">Upload your IoT metadata or try our sample data</p>
          </div>

          {!results && !analyzing && (
            <div className="bg-slate-800/50 backdrop-blur-lg border-2 border-dashed border-slate-600 rounded-2xl p-12">
              <div className="text-center">
                <Upload className="w-16 h-16 text-slate-400 mx-auto mb-6" />
                <h3 className="text-2xl font-bold text-white mb-4">Upload Your IoT Data</h3>
                <p className="text-slate-400 mb-6">
                  Upload a CSV file with your encrypted IoT metadata<br />
                  (device_type, encryption_algorithm, retention_period, etc.)
                </p>
                
                <div className="flex justify-center space-x-4">
                  <label className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold cursor-pointer transition-all inline-flex items-center space-x-2">
                    <Upload className="w-5 h-5" />
                    <span>Upload CSV</span>
                    <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
                  </label>
                  
                  <button
                    onClick={runDemo}
                    className="bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-lg font-semibold transition-all inline-flex items-center space-x-2"
                  >
                    <Zap className="w-5 h-5" />
                    <span>Try Sample Data</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {analyzing && (
            <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-2xl p-12">
              <div className="text-center">
                <div className="inline-block mb-6">
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-600 border-t-blue-500"></div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Analyzing Your Data...</h3>
                <div className="space-y-3 max-w-md mx-auto">
                  {['Extracting metadata features...', 'Running ML risk classifier...', 'Calculating migration strategies...', 'Generating recommendations...'].map((step, i) => (
                    <div key={i} className="flex items-center space-x-3 text-slate-300">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {results && (
            <div className="space-y-6">
              {/* Summary Cards */}
              <div className="grid grid-cols-3 gap-6">
                <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-slate-400 font-semibold">Total Records</h4>
                    <Database className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="text-4xl font-bold text-white">
                    {results?.totalRecords?.toLocaleString?.() ?? "—"}
                  </div>
                </div>
                
                <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-slate-400 font-semibold">Cost Savings</h4>
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  </div>
                  <div className="text-4xl font-bold text-green-400">
                     {results?.savings?.costSaving ?? 0}%
                  </div>

                  <div className="text-sm text-slate-400 mt-1">vs blanket migration</div>
                </div>
                
                <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-slate-400 font-semibold">Security Retained</h4>
                    <Shield className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="text-4xl font-bold text-blue-400">
                    {results?.savings?.securityRetained ?? 0}%
                  </div>

                  <div className="text-sm text-slate-400 mt-1">of maximum security</div>
                </div>
              </div>

              {/* Risk Distribution */}
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6">
                <h4 className="text-xl font-bold text-white mb-6 flex items-center space-x-2">
                  <BarChart3 className="w-6 h-6 text-blue-400" />
                  <span>Risk Distribution</span>
                </h4>
                <div className="grid grid-cols-4 gap-4">
                  {Object.entries(results?.riskDistribution ?? {}).map(([risk, count]) => {
                    const cnt = Number(count);
                    
                    return (
                      <div key={risk} className="text-center">
                        <div className={`text-3xl font-bold mb-2 ${getRiskColor(risk as RiskLevel)}`}>
                          {cnt.toLocaleString()}
                        </div>
                        <div className="text-sm text-slate-400 capitalize">{risk}</div>
                        <div className="text-xs text-slate-500 mt-1">
                          {(((cnt) / (results?.totalRecords ?? 1)) * 100).toFixed(1)}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Migration Recommendations */}
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6">
                <h4 className="text-xl font-bold text-white mb-6 flex items-center space-x-2">
                  <Lock className="w-6 h-6 text-purple-400" />
                  <span>Migration Strategy Breakdown</span>
                </h4>
                <div className="grid grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 border border-purple-500/30 rounded-lg p-4">
                    <div className="text-purple-400 text-sm font-semibold mb-2">Post-Quantum (Kyber768)</div>
                    <div className="text-3xl font-bold text-white mb-1">{results?.recommendations?.postquantum ?? 0}</div>
                    <div className="text-xs text-slate-400">High & Critical Risk</div>
                  </div>
                  <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 border border-blue-500/30 rounded-lg p-4">
                    <div className="text-blue-400 text-sm font-semibold mb-2">Hybrid (ECDH+Kyber)</div>
                    <div className="text-3xl font-bold text-white mb-1">{results?.recommendations?.hybrid ?? 0}</div>
                    <div className="text-xs text-slate-400">Medium Risk</div>
                  </div>
                  <div className="bg-gradient-to-br from-green-900/20 to-green-800/20 border border-green-500/30 rounded-lg p-4">
                    <div className="text-green-400 text-sm font-semibold mb-2">Keep Classical</div>
                    <div className="text-3xl font-bold text-white mb-1">{results?.recommendations?.classical ?? 0}</div>
                    <div className="text-xs text-slate-400">Low Risk</div>
                  </div>
                </div>
              </div>

              {/* Sample Predictions */}
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6">
                <h4 className="text-xl font-bold text-white mb-6 flex items-center space-x-2">
                  <FileText className="w-6 h-6 text-green-400" />
                  <span>Sample Device Recommendations</span>
                </h4>
                <div className="space-y-3">
                  {results?.predictions?.map((pred: any, i: number) => (
                    <div key={i} className="bg-slate-900/50 rounded-lg p-4 flex items-center justify-between">
                      <div className="flex-1">
                        <div className="font-semibold text-white mb-1">{pred.device}</div>
                        <div className="text-sm text-slate-400">
                          Current: <span className="text-orange-400">{pred.current}</span> | 
                          Retention: <span className="text-blue-400">{pred.retention}</span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${getRiskColor(pred.risk as RiskLevel)}`}>
                          {pred.risk.toUpperCase()}
                        </span>
                        <div className="text-right">
                          <div className="text-sm text-slate-400">Recommended:</div>
                          <div className="font-semibold text-green-400">{pred.recommended}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-center space-x-4">
                <button onClick={downloadReport} className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-semibold flex items-center space-x-2 transition-all">
                  <Download className="w-5 h-5" />
                  <span>Download Full Report (PDF)</span>
                </button>
                <button onClick={downloadMigrationPlan} className="bg-slate-700 hover:bg-slate-600 text-white px-8 py-4 rounded-xl font-semibold flex items-center space-x-2 transition-all">
                  <FileText className="w-5 h-5" />
                  <span>Export Migration Plan (CSV)</span>
                </button>
                <button 
                  onClick={() => {setResults(null); setUploadedFile(null); setShowDemo(false);}}
                  className="bg-slate-800 hover:bg-slate-700 text-white px-8 py-4 rounded-xl font-semibold transition-all"
                >
                  Analyze New Data
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Features Tab */}
      {activeTab === 'features' && (
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-white mb-4">Platform Features</h2>
            <p className="text-xl text-slate-300">Everything you need for quantum-safe IoT data migration</p>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-12">
            {[
              { title: 'ML-Powered Risk Assessment', desc: '85.8% accuracy in predicting quantum vulnerability using Random Forest classifier', icon: Brain, color: 'purple' },
              { title: 'Selective Migration Strategy', desc: 'Apply PQ only where needed—reduce overhead by 35% vs blanket migration', icon: Zap, color: 'blue' },
              { title: 'NIST-Standardized PQC', desc: 'Uses Kyber-768 (KEM) and Dilithium (signatures) approved by NIST', icon: Shield, color: 'green' },
              { title: 'Real-Time Analysis', desc: 'Process thousands of IoT records in minutes with GPU acceleration', icon: Cpu, color: 'orange' },
              { title: 'Comprehensive Reporting', desc: 'PDF reports, CSV exports, LaTeX tables for compliance documentation', icon: FileText, color: 'pink' },
              { title: 'API Integration', desc: 'REST API for seamless integration with your existing IoT infrastructure', icon: Server, color: 'cyan' }
            ].map((feature, i) => (
              <div key={i} className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 transition-all">
                <div
                    className={`bg-gradient-to-br ${
                      featureColorMap[feature.color as GradientColor]
                    } w-12 h-12 rounded-lg flex items-center justify-center mb-4`}
                  >
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-slate-400">{feature.desc}</p>
              </div>
            ))}
          </div>

          {/* Technical Specs */}
          <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-8">
            <h3 className="text-2xl font-bold text-white mb-6">Technical Specifications</h3>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-semibold text-blue-400 mb-3">Machine Learning</h4>
                <ul className="space-y-2 text-slate-300">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Random Forest classifier with 200 estimators</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>18 engineered features including quantum risk scores</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>85.8% test accuracy with 5-fold cross-validation</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>90%+ recall for high-risk data (security-first bias)</span>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="text-lg font-semibold text-purple-400 mb-3">Cryptography</h4>
                <ul className="space-y-2 text-slate-300">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Kyber-768 for post-quantum key encapsulation</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Dilithium for quantum-safe digital signatures</span>
</li>
<li className="flex items-start space-x-2">
  <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
  <span>Hybrid cryptography support (Classical + PQC)</span>
</li>
<li className="flex items-start space-x-2">
  <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
  <span>NIST-approved post-quantum algorithms</span>
</li>
<li className="flex items-start space-x-2">
  <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
  <span>Crypto-agility for future algorithm upgrades</span>
</li>
</ul>
</div>
</div>
</div>
</div>
)}
</div>
);
};

export default QuantumGuardAI;
