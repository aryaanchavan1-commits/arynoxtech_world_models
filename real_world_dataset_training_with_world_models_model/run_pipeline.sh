#!/bin/bash
# ===========================================================================
# AI4I Predictive Maintenance - Complete Pipeline
# Downloads data, trains world model, and evaluates anomaly detection
# ===========================================================================
# Usage: ./run_pipeline.sh
# ===========================================================================

set -e  # Exit on error

echo ""
echo "======================================================================"
echo "🏭 AI4I PREDICTIVE MAINTENANCE - WORLD MODEL TRAINING PIPELINE"
echo "======================================================================"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python version: $PYTHON_VERSION"

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch not found. Installing..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

python -c "import pandas; print(f'  Pandas: {pandas.__version__}')" 2>/dev/null || {
    echo "❌ Pandas not found. Installing..."
    pip install pandas
}

python -c "import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')" 2>/dev/null || {
    echo "❌ Matplotlib not found. Installing..."
    pip install matplotlib
}

python -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')" 2>/dev/null || {
    echo "❌ Scikit-learn not found. Installing..."
    pip install scikit-learn
}

echo "✅ All dependencies available"

# Step 1: Download dataset
echo ""
echo "======================================================================"
echo "📥 STEP 1: Downloading AI4I Dataset"
echo "======================================================================"
python download_datasets.py

# Step 2: Train world model
echo ""
echo "======================================================================"
echo "🚀 STEP 2: Training World Model on AI4I Data"
echo "======================================================================"
python train_ai4i.py --skip-download --epochs 50

# Step 3: Evaluate and detect anomalies
echo ""
echo "======================================================================"
echo "🔍 STEP 3: Evaluating & Detecting Anomalies"
echo "======================================================================"
python evaluate_ai4i.py

# Summary
echo ""
echo "======================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "📊 Results:"
echo "  - Training report:    models/ai4i_training_report.json"
echo "  - Training curves:    models/ai4i_training.png"
echo "  - Evaluation report:  models/evaluation/evaluation_report.json"
echo "  - Sensor plots:       models/evaluation/sensor_readings_with_anomalies.png"
echo "  - Error analysis:     models/evaluation/reconstruction_error_analysis.png"
echo "  - Precision-Recall:   models/evaluation/precision_recall_curve.png"
echo ""
echo "🎯 Next Steps:"
echo "  1. Deploy API:        python api.py"
echo "  2. Benchmark:         python benchmarks/benchmark.py"
echo "  3. View results:      open models/evaluation/sensor_readings_with_anomalies.png"
echo ""