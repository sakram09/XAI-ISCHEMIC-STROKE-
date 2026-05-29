# XAI-ISCHEMIC-STROKE-
#biotechnology #deep-learning #grad-cam #neuroscience #medical-imaging #explainable-ai #tensorflow #computer-vision
🧠 Explainable AI (XAI) for Ischemic Stroke Localization Using Grad-CAM & Deep Convolutional Neural Networks (CNN)📝 

Project Abstract: 
Ischemic stroke detection mein "Black-Box" AI models par trust karna mushkil hota hai. Is project ka maqsad sirf stroke detect karna nahi, balki Grad-CAM (Gradient-weighted Class Activation Mapping) ke zariye dimaag ke scans par stroke ki exact location ko highlight karna hai. Yeh model clinical interpretability ko behtar banata hai taake radiologists AI ke faisle ki wajah samajh saken.

🔬 Methodology & Mathematical Framework1. Grad-CAM Formulation Humne stroke localization ke liye Selvaraju et al. (2017) ki mathematical derivation use ki hai: 
Gradient Computation: Pehle hum output score ($y^c$) ka gradient nikalte hain feature map ($A^k$) ke mutabiq:$$\frac{\partial y^c}{\partial A^k_{ij}}$$Global Average Pooling: Phir hum importance weights ($\alpha^c_k$) calculate karte hain:$$\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$Heatmap Generation: Aakhir mein, weighted combination aur ReLU activation apply hota hai:$$L^c_{Grad-CAM} = ReLU\left(\sum_k \alpha^c_k A^k\right)$$2. CNN Architecture (VGG-Style)Architecture ko 4-block structure mein design kiya gaya hai taake feature extraction maximize ho sake:LayerTypeOutput ShapeActivationInputNeuro-Image$128 \times 128 \times 1$-Block 1Conv + MaxPool$64 \times 64 \times 32$ReLUBlock 2Conv + MaxPool$32 \times 32 \times 64$ReLUBlock 3Conv + MaxPool$16 \times 16 \times 128$ReLUBlock 4Grad-CAM Target$8 \times 8 \times 256$ReLUHeadGAP + Dense$1$ (Sigmoid)Sigmoid📊 Dataset & Pre-processingSource: Synthetic Neuro-Image Generation (Custom Engine).

Problem: Real MRI data sensitive hota hai aur easily accessible nahi hota.Solution: Humne aik pipeline banayi jo random Gaussian noise aur lesion injection (Infarct zones) ke zariye high-fidelity brain slices generate karti hai.Augmentation: Normalization, Rescaling, aur Random spatial shifts use kiye gaye hain.🛠️ Technical StackFramework: TensorFlow 2.x / KerasLibraries: * NumPy: Data manipulation aur matrix operations ke liye.OpenCV: Image processing aur heatmap overlays ke liye.Matplotlib & Seaborn: Visualization aur results plotting ke liye.Techniques: GradientTape (for custom gradients), Global Average Pooling (GAP), Dropout (for regularization).

🚀 Key Features: Automated Diagnosis: High-accuracy stroke vs normal classification.Visual Explanations: Heatmaps jo batate hain ke AI dimaag ke kis hisse ko "dekh" raha hai.Clinical Reports: ASCII-based radiology reports jo diagnosis aur confidence level print karti hain.Quantitative Metrics: Pointing error (pixels mein) aur ROI activation ratio ka analysis.
📈 Quantitative ResultsAvg Pointing Error: ~12 pixels.Localization Confidence: 94% - 98%.Inference Time: < 50ms per scan (on T4 GPU).📁 Repository StructurePlaintext├── Explainable_Stroke_Detection.ipynb  # Main Project Notebook
├── stroke_detection_model.h5           # Pre-trained Weights
├── research_results/                   # Saved Dashboards & Reports
└── README.md                           # Documentation
How to Use? Open the .ipynb file in Google Colab. Change the runtime to T4 GPU.
Run all cells to generate live heatmaps and clinical reports. 
Future Scope: Integrating SHAP (Shapley Additive exPlanations) for pixel-level attribution.
Testing on the ISLES 2015 (Ischemic Stroke Lesion Segmentation) public dataset.
