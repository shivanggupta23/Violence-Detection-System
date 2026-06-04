# """
# ═══════════════════════════════════════════════════════════
# FIXED VIOLENCE DETECTION STREAMLIT APP
# Bug Fix: Exact feature extraction matching training
# ═══════════════════════════════════════════════════════════
# """

# import streamlit as st
# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from PIL import Image
# import open_clip
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import tempfile
# import os
# import matplotlib.pyplot as plt
# import pandas as pd

# # ═══════════════════════════════════════════════════════════
# # PAGE CONFIG
# # ═══════════════════════════════════════════════════════════

# st.set_page_config(
#     page_title="Violence Detection System",
#     page_icon="🚨",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ═══════════════════════════════════════════════════════════
# # CUSTOM CSS
# # ═══════════════════════════════════════════════════════════

# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         color: #FF4B4B;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #FF4B4B;
#     }
#     .violence-detected {
#         background-color: #ffebee;
#         border-left: 4px solid #f44336;
#     }
#     .safe-detected {
#         background-color: #e8f5e9;
#         border-left: 4px solid #4caf50;
#     }
#     .debug-box {
#         background-color: #fff3cd;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #ffc107;
#         margin: 1rem 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # MODEL DEFINITIONS
# # ═══════════════════════════════════════════════════════════

# class ViolenceAgent(nn.Module):
#     """Violence Detection Model"""
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(ViolenceAgent, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # ═══════════════════════════════════════════════════════════
# # LOAD MODELS (CACHED)
# # ═══════════════════════════════════════════════════════════

# @st.cache_resource
# def load_models():
#     """Load all required models (cached for performance)"""
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # 1. Load Violence Detection Model
#     violence_model = ViolenceAgent(input_dim=1024, hidden1=512, hidden2=256, dropout=0.3)
    
#     model_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env\models\violence_agent_best.pth"
#     violence_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     violence_model = violence_model.to(device)
#     violence_model.eval()
    
#     # 2. Load OpenCLIP (EXACT SAME AS TRAINING)
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32', 
#         pretrained='laion2b_s34b_b79k'
#     )
#     clip_model = clip_model.to(device)
#     clip_model.eval()
    
#     # 3. Load OpenCLIP tokenizer
#     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
#     # 4. Load BLIP for captions
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained(
#         "Salesforce/blip-image-captioning-base"
#     ).to(device)
#     blip_model.eval()
    
#     st.sidebar.success(f"✅ Models loaded on {device.upper()}")
    
#     return {
#         'violence_model': violence_model,
#         'clip_model': clip_model,
#         'preprocess': preprocess,
#         'tokenizer': tokenizer,
#         'blip_processor': blip_processor,
#         'blip_model': blip_model,
#         'device': device
#     }

# # ═══════════════════════════════════════════════════════════
# # VIDEO PROCESSING FUNCTIONS
# # ═══════════════════════════════════════════════════════════

# def extract_frames(video_path, fps=1):
#     """Extract frames from video at specified FPS"""
    
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             # Convert BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
#     return frames

# def generate_caption(image, blip_processor, blip_model, device):
#     """Generate caption using BLIP (EXACT MATCH TO TRAINING)"""
    
#     # Convert to PIL
#     pil_image = Image.fromarray(image.astype('uint8'))
    
#     # Process (same as training)
#     inputs = blip_processor(pil_image, return_tensors="pt").to(device)
    
#     # Generate caption
#     with torch.no_grad():
#         out = blip_model.generate(**inputs, max_length=50)
    
#     caption = blip_processor.decode(out[0], skip_special_tokens=True)
#     return caption

# def get_image_embedding(image, clip_model, preprocess, device):
#     """
#     Get OpenCLIP image embedding
    
#     ⚠️ CRITICAL: EXACT MATCH TO TRAINING
#     """
    
#     # Convert to PIL
#     pil_image = Image.fromarray(image.astype('uint8'))
    
#     # Preprocess (same as training)
#     image_input = preprocess(pil_image).unsqueeze(0).to(device)
    
#     # Extract features
#     with torch.no_grad():
#         # ⚠️ CRITICAL: Use encode_image (not forward)
#         image_features = clip_model.encode_image(image_input)
        
#         # ⚠️ CRITICAL: Normalize (MUST match training)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
#     # Convert to numpy
#     embedding = image_features.cpu().numpy()[0]
    
#     # Verify shape
#     assert embedding.shape[0] == 512, f"Expected 512-d, got {embedding.shape[0]}"
    
#     return embedding

# def get_text_embedding(caption, clip_model, tokenizer, device):
#     """
#     Get OpenCLIP text embedding
    
#     ⚠️ CRITICAL: EXACT MATCH TO TRAINING
#     """
    
#     # Tokenize (same as training)
#     text_input = tokenizer([caption]).to(device)
    
#     # Extract features
#     with torch.no_grad():
#         # ⚠️ CRITICAL: Use encode_text (not forward)
#         text_features = clip_model.encode_text(text_input)
        
#         # ⚠️ CRITICAL: Normalize (MUST match training)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
#     # Convert to numpy
#     embedding = text_features.cpu().numpy()[0]
    
#     # Verify shape
#     assert embedding.shape[0] == 512, f"Expected 512-d, got {embedding.shape[0]}"
    
#     return embedding

# def predict_violence(combined_features, violence_model, device):
#     """
#     Predict violence with confidence
    
#     ⚠️ CRITICAL: Model in eval mode
#     """
    
#     # Ensure model is in eval mode
#     violence_model.eval()
    
#     # Convert to tensor
#     if isinstance(combined_features, np.ndarray):
#         features_tensor = torch.FloatTensor(combined_features)
#     else:
#         features_tensor = combined_features
    
#     # Add batch dimension if needed
#     if features_tensor.dim() == 1:
#         features_tensor = features_tensor.unsqueeze(0)
    
#     features_tensor = features_tensor.to(device)
    
#     # Predict
#     with torch.no_grad():
#         outputs = violence_model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     # Extract results
#     pred_class = int(np.argmax(probs))
#     confidence = float(probs[pred_class])
#     violence_prob = float(probs[1])
#     safe_prob = float(probs[0])
    
#     return pred_class, confidence, probs, violence_prob, safe_prob

# # ═══════════════════════════════════════════════════════════
# # COMPARISON WITH TRAINING FEATURES
# # ═══════════════════════════════════════════════════════════

# def load_training_feature_sample():
#     """Load a sample feature from training data for comparison"""
    
#     try:
#         # Try to load a violence feature from training set
#         violence_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env\combined_features\test\violence"
        
#         if os.path.exists(violence_path):
#             videos = [v for v in os.listdir(violence_path) if os.path.isdir(os.path.join(violence_path, v))]
#             if videos:
#                 video_path = os.path.join(violence_path, videos[0])
#                 features = [f for f in os.listdir(video_path) if f.endswith('.npy')]
#                 if features:
#                     feature_path = os.path.join(video_path, features[0])
#                     feature = np.load(feature_path)
#                     return feature, f"{videos[0]}/{features[0]}"
#     except Exception as e:
#         st.error(f"Could not load training feature: {e}")
    
#     return None, None

# # ═══════════════════════════════════════════════════════════
# # STREAMLIT APP
# # ═══════════════════════════════════════════════════════════

# def main():
    
#     # Header
#     st.markdown('<div class="main-header">🚨 Violence Detection System</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sub-header">Multimodal AI for Real-Time Violence Detection</div>', unsafe_allow_html=True)
    
#     # Sidebar
#     st.sidebar.title("⚙️ Configuration")
#     st.sidebar.markdown("---")
    
#     # Model info
#     with st.sidebar.expander("ℹ️ About the Model"):
#         st.write("""
#         **Multimodal Violence Detection**
        
#         - **Accuracy:** 99.90%
#         - **Architecture:** Vision-Language Fusion
#         - **Features:** 
#           - OpenCLIP image embeddings (512-d)
#           - BLIP caption generation
#           - OpenCLIP text embeddings (512-d)
#           - MLP classifier (1024→512→256→2)
#         """)
    
#     # Settings
#     fps = st.sidebar.slider("Frames per second", 1, 5, 1)
#     confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
    
#     # Debug mode
#     debug_mode = st.sidebar.checkbox("🔍 Debug Mode (show feature details)", value=True)
    
#     # Test with training data
#     if st.sidebar.button("🧪 Test with Training Data"):
#         with st.spinner("Loading training feature..."):
#             models = load_models()
#             training_feature, feature_name = load_training_feature_sample()
            
#             if training_feature is not None:
#                 st.sidebar.success(f"Loaded: {feature_name}")
                
#                 # Predict on training feature
#                 pred_class, confidence, probs, v_prob, s_prob = predict_violence(
#                     training_feature,
#                     models['violence_model'],
#                     models['device']
#                 )
                
#                 st.sidebar.markdown("**Training Feature Test:**")
#                 st.sidebar.write(f"Prediction: {'Violence' if pred_class == 1 else 'Safe'}")
#                 st.sidebar.write(f"Confidence: {confidence:.4f}")
#                 st.sidebar.write(f"Violence Prob: {v_prob:.4f}")
                
#                 if pred_class == 1:
#                     st.sidebar.success("✅ Correctly predicts violence!")
#                 else:
#                     st.sidebar.error("❌ Should be violence but predicted safe!")
    
#     st.sidebar.markdown("---")
#     st.sidebar.info("Upload a video to detect violence frame-by-frame")
    
#     # Main content
#     tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "📖 Documentation"])
    
#     with tab1:
#         # File upload
#         uploaded_file = st.file_uploader(
#             "Choose a video file",
#             type=['mp4', 'avi', 'mov', 'mkv'],
#             help="Upload a video file to analyze"
#         )
        
#         if uploaded_file is not None:
            
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 video_path = tmp_file.name
            
#             # Display video
#             st.video(uploaded_file)
            
#             # Process button
#             if st.button("🔍 Analyze Video", type="primary"):
                
#                 # Load models
#                 with st.spinner("Loading AI models..."):
#                     models = load_models()
                
#                 st.info(f"🖥️ Running on: {models['device'].upper()}")
                
#                 # Extract frames
#                 with st.spinner(f"Extracting frames at {fps} FPS..."):
#                     frames = extract_frames(video_path, fps=fps)
#                     st.success(f"✅ Extracted {len(frames)} frames")
                
#                 # Process each frame
#                 results = []
                
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 # Debug: Show feature extraction for first frame
#                 if debug_mode and len(frames) > 0:
#                     st.markdown("---")
#                     st.subheader("🔍 Debug: First Frame Feature Extraction")
                    
#                     debug_frame = frames[0]
                    
#                     col1, col2 = st.columns([1, 2])
                    
#                     with col1:
#                         st.image(debug_frame, caption="First Frame", use_container_width=True)
                    
#                     with col2:
#                         with st.spinner("Extracting features..."):
                            
#                             # Generate caption
#                             debug_caption = generate_caption(
#                                 debug_frame,
#                                 models['blip_processor'],
#                                 models['blip_model'],
#                                 models['device']
#                             )
                            
#                             # Get embeddings
#                             debug_img_emb = get_image_embedding(
#                                 debug_frame,
#                                 models['clip_model'],
#                                 models['preprocess'],
#                                 models['device']
#                             )
                            
#                             debug_text_emb = get_text_embedding(
#                                 debug_caption,
#                                 models['clip_model'],
#                                 models['tokenizer'],
#                                 models['device']
#                             )
                            
#                             # Combine
#                             debug_combined = np.concatenate([debug_img_emb, debug_text_emb])
                            
#                             # Predict
#                             debug_pred, debug_conf, _, debug_v_prob, debug_s_prob = predict_violence(
#                                 debug_combined,
#                                 models['violence_model'],
#                                 models['device']
#                             )
                            
#                             st.markdown(f"""
#                             <div class="debug-box">
#                             <strong>🎯 Caption:</strong> {debug_caption}<br>
#                             <strong>📐 Image Embedding:</strong> shape={debug_img_emb.shape}, mean={debug_img_emb.mean():.4f}, norm={np.linalg.norm(debug_img_emb):.4f}<br>
#                             <strong>📝 Text Embedding:</strong> shape={debug_text_emb.shape}, mean={debug_text_emb.mean():.4f}, norm={np.linalg.norm(debug_text_emb):.4f}<br>
#                             <strong>🔗 Combined:</strong> shape={debug_combined.shape}, mean={debug_combined.mean():.4f}<br>
#                             <br>
#                             <strong>🎲 Prediction:</strong> {'Violence' if debug_pred == 1 else 'Safe'}<br>
#                             <strong>💯 Confidence:</strong> {debug_conf:.4f}<br>
#                             <strong>📊 Probabilities:</strong> Violence={debug_v_prob:.4f}, Safe={debug_s_prob:.4f}
#                             </div>
#                             """, unsafe_allow_html=True)
                            
#                             # Compare with training feature
#                             training_feature, feature_name = load_training_feature_sample()
#                             if training_feature is not None:
#                                 st.markdown("**Comparison with Training Feature:**")
                                
#                                 train_img = training_feature[:512]
#                                 train_text = training_feature[512:]
                                
#                                 img_diff = np.abs(debug_img_emb - train_img).mean()
#                                 text_diff = np.abs(debug_text_emb - train_text).mean()
                                
#                                 st.write(f"- Image embedding difference: {img_diff:.6f}")
#                                 st.write(f"- Text embedding difference: {text_diff:.6f}")
                                
#                                 if img_diff > 0.5 or text_diff > 0.5:
#                                     st.warning("⚠️ Large difference from training features!")
#                                 else:
#                                     st.success("✅ Features similar to training data")
                    
#                     st.markdown("---")
                
#                 # Process all frames
#                 for idx, frame in enumerate(frames):
                    
#                     status_text.text(f"Processing frame {idx+1}/{len(frames)}...")
                    
#                     try:
#                         # 1. Generate caption
#                         caption = generate_caption(
#                             frame,
#                             models['blip_processor'],
#                             models['blip_model'],
#                             models['device']
#                         )
                        
#                         # 2. Get image embedding
#                         img_embedding = get_image_embedding(
#                             frame,
#                             models['clip_model'],
#                             models['preprocess'],
#                             models['device']
#                         )
                        
#                         # 3. Get text embedding
#                         text_embedding = get_text_embedding(
#                             caption,
#                             models['clip_model'],
#                             models['tokenizer'],
#                             models['device']
#                         )
                        
#                         # 4. Combine features (EXACT SAME AS TRAINING)
#                         combined_features = np.concatenate([img_embedding, text_embedding])
                        
#                         # Verify
#                         assert combined_features.shape[0] == 1024, f"Wrong shape: {combined_features.shape}"
                        
#                         # 5. Predict
#                         pred_class, confidence, probs, violence_prob, safe_prob = predict_violence(
#                             combined_features,
#                             models['violence_model'],
#                             models['device']
#                         )
                        
#                         # Store result
#                         results.append({
#                             'frame_number': idx,
#                             'timestamp': f"{idx/fps:.1f}s",
#                             'prediction': 'Violence' if pred_class == 1 else 'Safe',
#                             'confidence': confidence,
#                             'violence_prob': violence_prob,
#                             'safe_prob': safe_prob,
#                             'caption': caption,
#                             'frame': frame,
#                             'img_emb_norm': float(np.linalg.norm(img_embedding)),
#                             'text_emb_norm': float(np.linalg.norm(text_embedding))
#                         })
                        
#                     except Exception as e:
#                         st.error(f"Error processing frame {idx}: {str(e)}")
#                         import traceback
#                         st.code(traceback.format_exc())
                        
#                         results.append({
#                             'frame_number': idx,
#                             'timestamp': f"{idx/fps:.1f}s",
#                             'prediction': 'Error',
#                             'confidence': 0.0,
#                             'violence_prob': 0.0,
#                             'safe_prob': 0.0,
#                             'caption': f"Error: {str(e)}",
#                             'frame': frame,
#                             'img_emb_norm': 0.0,
#                             'text_emb_norm': 0.0
#                         })
                    
#                     progress_bar.progress((idx + 1) / len(frames))
                
#                 status_text.text("✅ Analysis complete!")
                
#                 # Show summary
#                 violence_count = sum(1 for r in results if r['prediction'] == 'Violence')
#                 safe_count = sum(1 for r in results if r['prediction'] == 'Safe')
                
#                 if violence_count > 0:
#                     st.success(f"🚨 **VIOLENCE DETECTED!** {violence_count} frames contain violence")
#                 else:
#                     st.info(f"✅ **NO VIOLENCE DETECTED.** All {safe_count} frames are safe.")
                
#                 if violence_count == 0 and safe_count > 0:
#                     st.warning("""
#                     ⚠️ **Debugging Tips:**
#                     - Check if video contains REAL violence (not martial arts/gym training)
#                     - Review debug info above
#                     - Compare feature norms with training data
#                     - Test with a known violence video from training set
#                     """)
                
#                 # Store in session
#                 st.session_state['results'] = results
#                 st.session_state['video_analyzed'] = True
                
#                 # Clean up
#                 try:
#                     os.unlink(video_path)
#                 except:
#                     pass
    
#     with tab2:
#         if 'video_analyzed' in st.session_state and st.session_state['video_analyzed']:
            
#             results = st.session_state['results']
            
#             # Summary
#             st.subheader("📊 Analysis Summary")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             total_frames = len(results)
#             violence_frames = sum(1 for r in results if r['prediction'] == 'Violence')
#             safe_frames = sum(1 for r in results if r['prediction'] == 'Safe')
#             avg_confidence = np.mean([r['confidence'] for r in results])
            
#             col1.metric("Total Frames", total_frames)
#             col2.metric("Violence Detected", violence_frames, delta=f"{violence_frames/total_frames*100:.1f}%")
#             col3.metric("Safe Frames", safe_frames, delta=f"{safe_frames/total_frames*100:.1f}%")
#             col4.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
#             # Timeline
#             st.subheader("📈 Detection Timeline")
            
#             df = pd.DataFrame(results)
            
#             fig, ax = plt.subplots(figsize=(12, 4))
            
#             colors = ['red' if p == 'Violence' else 'green' for p in df['prediction']]
#             ax.scatter(df['frame_number'], df['confidence'], c=colors, alpha=0.6, s=100)
#             ax.axhline(y=confidence_threshold, color='orange', linestyle='--', label=f'Threshold ({confidence_threshold})')
#             ax.set_xlabel('Frame Number', fontsize=12)
#             ax.set_ylabel('Confidence', fontsize=12)
#             ax.set_title('Violence Detection Over Time', fontsize=14, fontweight='bold')
#             ax.legend()
#             ax.grid(alpha=0.3)
#             ax.set_ylim([0, 1])
            
#             st.pyplot(fig)
            
#             # Frame results
#             st.subheader("🎬 Frame-by-Frame Analysis")
            
#             # Filter
#             filter_option = st.selectbox(
#                 "Show frames:",
#                 ["All Frames", "Violence Only", "Safe Only", "High Confidence (>90%)"]
#             )
            
#             filtered_results = results
#             if filter_option == "Violence Only":
#                 filtered_results = [r for r in results if r['prediction'] == 'Violence']
#             elif filter_option == "Safe Only":
#                 filtered_results = [r for r in results if r['prediction'] == 'Safe']
#             elif filter_option == "High Confidence (>90%)":
#                 filtered_results = [r for r in results if r['confidence'] > 0.9]
            
#             st.info(f"Showing {len(filtered_results)} of {len(results)} frames")
            
#             # Display grid
#             cols_per_row = 3
#             for i in range(0, len(filtered_results), cols_per_row):
#                 cols = st.columns(cols_per_row)
                
#                 for j in range(cols_per_row):
#                     if i + j < len(filtered_results):
#                         result = filtered_results[i + j]
                        
#                         with cols[j]:
#                             st.image(result['frame'], use_container_width=True)
                            
#                             card_class = "violence-detected" if result['prediction'] == 'Violence' else "safe-detected"
                            
#                             st.markdown(f"""
#                                 <div class="metric-card {card_class}">
#                                     <strong>Frame {result['frame_number']}</strong> ({result['timestamp']})<br>
#                                     <strong>Prediction:</strong> {result['prediction']}<br>
#                                     <strong>Confidence:</strong> {result['confidence']:.1%}<br>
#                                     <strong>Violence Prob:</strong> {result['violence_prob']:.1%}<br>
#                                     <strong>Safe Prob:</strong> {result['safe_prob']:.1%}<br>
#                                     <strong>Caption:</strong> {result['caption']}
#                                 </div>
#                             """, unsafe_allow_html=True)
            
#             # Export
#             st.subheader("💾 Export Results")
            
#             export_df = pd.DataFrame([{
#                 'Frame': r['frame_number'],
#                 'Timestamp': r['timestamp'],
#                 'Prediction': r['prediction'],
#                 'Confidence': f"{r['confidence']:.4f}",
#                 'Violence Probability': f"{r['violence_prob']:.4f}",
#                 'Safe Probability': f"{r['safe_prob']:.4f}",
#                 'Caption': r['caption'],
#                 'Image_Emb_Norm': f"{r['img_emb_norm']:.4f}",
#                 'Text_Emb_Norm': f"{r['text_emb_norm']:.4f}"
#             } for r in results])
            
#             csv = export_df.to_csv(index=False)
            
#             st.download_button(
#                 label="📥 Download Results (CSV)",
#                 data=csv,
#                 file_name="violence_detection_results.csv",
#                 mime="text/csv"
#             )
        
#         else:
#             st.info("👆 Upload and analyze a video in the 'Upload & Analyze' tab first")
    
#     with tab3:
#         st.subheader("📖 Documentation")
        
#         st.markdown("""
#         ### 🧠 How It Works
        
#         1. **Frame Extraction:** Extract frames at specified FPS
#         2. **Image Features:** OpenCLIP ViT-B/32 (512-d, normalized)
#         3. **Caption Generation:** BLIP describes the scene
#         4. **Text Features:** OpenCLIP text encoder (512-d, normalized)
#         5. **Fusion:** Concatenate image + text → 1024-d
#         6. **Classification:** MLP (1024→512→256→2) → Violence/Safe
        
#         ### 📊 Model Performance
        
#         - Test Accuracy: 99.90%
#         - ROC AUC: 0.9998
#         - Calibration ECE: 0.0011
        
#         ### 🔍 Debugging
        
#         If no violence detected in violence video:
#         1. Enable Debug Mode (sidebar)
#         2. Check feature norms (should be ~1.0)
#         3. Compare with training features
#         4. Test with training data button
#         5. Verify video contains REAL violence (not gym/martial arts)
#         """)

# if __name__ == "__main__":
#     main()




# """
# ═══════════════════════════════════════════════════════════
# VIOLENCE DETECTION STREAMLIT - FINAL FIX
# Matches training feature extraction EXACTLY
# ═══════════════════════════════════════════════════════════
# """

# import streamlit as st
# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from PIL import Image
# import open_clip
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import tempfile
# import os
# import matplotlib.pyplot as plt
# import pandas as pd

# # ═══════════════════════════════════════════════════════════
# # PAGE CONFIG
# # ═══════════════════════════════════════════════════════════

# st.set_page_config(
#     page_title="Violence Detection System",
#     page_icon="🚨",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ═══════════════════════════════════════════════════════════
# # MODEL DEFINITION
# # ═══════════════════════════════════════════════════════════

# class ViolenceAgent(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(ViolenceAgent, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # ═══════════════════════════════════════════════════════════
# # LOAD MODELS
# # ═══════════════════════════════════════════════════════════

# @st.cache_resource
# def load_models():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Violence model
#     violence_model = ViolenceAgent()
#     model_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env\models\violence_agent_best.pth"
#     violence_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     violence_model = violence_model.to(device)
#     violence_model.eval()
    
#     # OpenCLIP
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32',
#         pretrained='laion2b_s34b_b79k'
#     )
#     clip_model = clip_model.to(device)
#     clip_model.eval()
    
#     # Tokenizer
#     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
#     # BLIP
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained(
#         "Salesforce/blip-image-captioning-base"
#     ).to(device)
#     blip_model.eval()
    
#     return {
#         'violence_model': violence_model,
#         'clip_model': clip_model,
#         'preprocess': preprocess,
#         'tokenizer': tokenizer,
#         'blip_processor': blip_processor,
#         'blip_model': blip_model,
#         'device': device
#     }

# # ═══════════════════════════════════════════════════════════
# # FEATURE EXTRACTION (EXACT MATCH TO TRAINING)
# # ═══════════════════════════════════════════════════════════

# def extract_features_from_frame(frame, models):
#     """
#     Extract features EXACTLY as done during training
    
#     Returns: 1024-d combined feature vector
#     """
    
#     device = models['device']
    
#     # Convert numpy array to PIL Image
#     if isinstance(frame, np.ndarray):
#         pil_image = Image.fromarray(frame.astype('uint8'))
#     else:
#         pil_image = frame
    
#     # ═══════════════════════════════════════════════════════
#     # STEP 1: IMAGE EMBEDDING (OpenCLIP)
#     # ═══════════════════════════════════════════════════════
    
#     # Preprocess image (resizes to 224x224, normalizes)
#     image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         # Extract image features
#         image_features = models['clip_model'].encode_image(image_input)
        
#         # CRITICAL: Normalize to unit length
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
#     image_embedding = image_features.cpu().numpy()[0]
    
#     # ═══════════════════════════════════════════════════════
#     # STEP 2: CAPTION GENERATION (BLIP)
#     # ═══════════════════════════════════════════════════════
    
#     # Generate caption
#     blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
#     caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
#     # ═══════════════════════════════════════════════════════
#     # STEP 3: TEXT EMBEDDING (OpenCLIP)
#     # ═══════════════════════════════════════════════════════
    
#     # Tokenize caption
#     text_input = models['tokenizer']([caption]).to(device)
    
#     with torch.no_grad():
#         # Extract text features
#         text_features = models['clip_model'].encode_text(text_input)
        
#         # CRITICAL: Normalize to unit length
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
#     text_embedding = text_features.cpu().numpy()[0]
    
#     # ═══════════════════════════════════════════════════════
#     # STEP 4: COMBINE FEATURES
#     # ═══════════════════════════════════════════════════════
    
#     # Concatenate image + text
#     combined_features = np.concatenate([image_embedding, text_embedding])
    
#     # Verify dimensions
#     assert image_embedding.shape[0] == 512, f"Image embedding wrong size: {image_embedding.shape}"
#     assert text_embedding.shape[0] == 512, f"Text embedding wrong size: {text_embedding.shape}"
#     assert combined_features.shape[0] == 1024, f"Combined wrong size: {combined_features.shape}"
    
#     # Verify normalization
#     img_norm = np.linalg.norm(image_embedding)
#     text_norm = np.linalg.norm(text_embedding)
    
#     return {
#         'combined_features': combined_features,
#         'image_embedding': image_embedding,
#         'text_embedding': text_embedding,
#         'caption': caption,
#         'img_norm': float(img_norm),
#         'text_norm': float(text_norm)
#     }

# def predict_violence(combined_features, model, device):
#     """Predict violence from combined features"""
    
#     model.eval()
    
#     # Convert to tensor
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     pred_class = int(np.argmax(probs))
#     confidence = float(probs[pred_class])
    
#     return {
#         'prediction': pred_class,
#         'confidence': confidence,
#         'violence_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# # ═══════════════════════════════════════════════════════════
# # VIDEO PROCESSING
# # ═══════════════════════════════════════════════════════════

# def extract_frames(video_path, fps=1):
#     """Extract frames from video"""
    
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
#     return frames

# # ═══════════════════════════════════════════════════════════
# # STREAMLIT UI
# # ═══════════════════════════════════════════════════════════

# def main():
    
#     st.title("🚨 Violence Detection System")
#     st.markdown("*Multimodal AI for Real-Time Violence Detection*")
    
#     # Sidebar
#     st.sidebar.title("⚙️ Settings")
#     fps = st.sidebar.slider("Frames per second", 1, 5, 1)
    
#     # Test with training data
#     if st.sidebar.button("🧪 Test Model"):
#         with st.spinner("Testing model..."):
#             models = load_models()
            
#             # Load a training violence feature
#             violence_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env\combined_features\test\violence"
            
#             if os.path.exists(violence_path):
#                 videos = [v for v in os.listdir(violence_path) if os.path.isdir(os.path.join(violence_path, v))]
#                 if videos:
#                     video_path = os.path.join(violence_path, videos[0])
#                     features = [f for f in os.listdir(video_path) if f.endswith('.npy')]
#                     if features:
#                         feature_path = os.path.join(video_path, features[0])
#                         training_feature = np.load(feature_path)
                        
#                         result = predict_violence(
#                             training_feature,
#                             models['violence_model'],
#                             models['device']
#                         )
                        
#                         if result['prediction'] == 1:
#                             st.sidebar.success(f"✅ Model works! Detected violence with {result['confidence']:.1%} confidence")
#                         else:
#                             st.sidebar.error("❌ Model failed! Should detect violence")
    
#     # Main upload area
#     uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
#     if uploaded_file:
        
#         # Save temp file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             video_path = tmp_file.name
        
#         st.video(uploaded_file)
        
#         if st.button("🔍 Analyze", type="primary"):
            
#             # Load models
#             with st.spinner("Loading models..."):
#                 models = load_models()
            
#             # Extract frames
#             with st.spinner(f"Extracting frames at {fps} FPS..."):
#                 frames = extract_frames(video_path, fps)
#                 st.success(f"✅ Extracted {len(frames)} frames")
            
#             # Process frames
#             results = []
#             progress = st.progress(0)
            
#             for idx, frame in enumerate(frames):
                
#                 # Extract features
#                 feature_data = extract_features_from_frame(frame, models)
                
#                 # Predict
#                 prediction = predict_violence(
#                     feature_data['combined_features'],
#                     models['violence_model'],
#                     models['device']
#                 )
                
#                 # Store result
#                 results.append({
#                     'frame': idx,
#                     'timestamp': f"{idx/fps:.1f}s",
#                     'prediction': 'Violence' if prediction['prediction'] == 1 else 'Safe',
#                     'confidence': prediction['confidence'],
#                     'violence_prob': prediction['violence_prob'],
#                     'safe_prob': prediction['safe_prob'],
#                     'caption': feature_data['caption'],
#                     'img_norm': feature_data['img_norm'],
#                     'text_norm': feature_data['text_norm'],
#                     'frame_image': frame
#                 })
                
#                 progress.progress((idx + 1) / len(frames))
            
#             # Summary
#             violence_count = sum(1 for r in results if r['prediction'] == 'Violence')
            
#             st.markdown("---")
            
#             if violence_count > 0:
#                 st.error(f"🚨 **VIOLENCE DETECTED!** {violence_count}/{len(frames)} frames")
#             else:
#                 st.success(f"✅ **NO VIOLENCE DETECTED** in {len(frames)} frames")
            
#             # Metrics
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("Total Frames", len(frames))
#             col2.metric("Violence", violence_count)
#             col3.metric("Safe", len(frames) - violence_count)
#             col4.metric("Avg Confidence", f"{np.mean([r['confidence'] for r in results]):.1%}")
            
#             # Timeline
#             st.subheader("📈 Detection Timeline")
            
#             fig, ax = plt.subplots(figsize=(12, 4))
            
#             colors = ['red' if r['prediction'] == 'Violence' else 'green' for r in results]
#             ax.scatter(range(len(results)), [r['confidence'] for r in results], c=colors, s=100, alpha=0.6)
#             ax.set_xlabel('Frame Number')
#             ax.set_ylabel('Confidence')
#             ax.set_title('Violence Detection Over Time')
#             ax.grid(alpha=0.3)
            
#             st.pyplot(fig)
            
#             # Frame details
            
#             # Frame details
#             st.subheader("🎬 Frame Details")
#             show_all = st.checkbox("Show all frames", value=False, key="show_all_frames")

#             for i, result in enumerate(results):
#                 if result['prediction'] == 'Violence' or show_all:
#                     with st.expander(f"Frame {i} - {result['timestamp']} - {result['prediction']}"):
#                         col1, col2 = st.columns([1, 2])
#                         with col1:
#                             st.image(result['frame_image'], use_container_width=True)
#                         with col2:
#                             st.write(f"**Prediction:** {result['prediction']}")
#                             st.write(f"**Confidence:** {result['confidence']:.1%}")
#                             st.write(f"**Violence Prob:** {result['violence_prob']:.1%}")
#                             st.write(f"**Safe Prob:** {result['safe_prob']:.1%}")
#                             st.write(f"**Caption:** {result['caption']}")
#                             st.write(f"**Image Norm:** {result['img_norm']:.4f} (should be ~1.0)")
#                             st.write(f"**Text Norm:** {result['text_norm']:.4f} (should be ~1.0)")
#             # Export
#             st.subheader("💾 Export Results")
            
#             df = pd.DataFrame([{
#                 'Frame': r['frame'],
#                 'Timestamp': r['timestamp'],
#                 'Prediction': r['prediction'],
#                 'Confidence': f"{r['confidence']:.4f}",
#                 'Violence_Prob': f"{r['violence_prob']:.4f}",
#                 'Safe_Prob': f"{r['safe_prob']:.4f}",
#                 'Caption': r['caption']
#             } for r in results])
            
#             csv = df.to_csv(index=False)
            
#             st.download_button(
#                 "📥 Download CSV",
#                 csv,
#                 "violence_detection_results.csv",
#                 "text/csv"
#             )
            
#             # Cleanup
#             try:
#                 os.unlink(video_path)
#             except:
#                 pass

# if __name__ == "__main__":
#     main()





"""
═══════════════════════════════════════════════════════════
VIOLENCE DETECTION STREAMLIT - FINAL WORKING VERSION
NO NORMALIZATION (matches training)
═══════════════════════════════════════════════════════════
"""

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Violence Detection System",
    page_icon="🚨",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════

class ViolenceAgent(nn.Module):
    def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
        super(ViolenceAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Violence model
    violence_model = ViolenceAgent()
    model_path = "models/violence_agent_best.pth"
    violence_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    violence_model = violence_model.to(device)
    violence_model.eval()
    
    # OpenCLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Tokenizer
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # BLIP
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    blip_model.eval()
    
    return {
        'violence_model': violence_model,
        'clip_model': clip_model,
        'preprocess': preprocess,
        'tokenizer': tokenizer,
        'blip_processor': blip_processor,
        'blip_model': blip_model,
        'device': device
    }

# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION (NO NORMALIZATION!)
# ═══════════════════════════════════════════════════════════

def extract_features_from_frame(frame, models):
    """Extract features WITHOUT normalization (matches training)"""
    
    device = models['device']
    
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(frame.astype('uint8'))
    else:
        pil_image = frame
    
    # Image embedding (NO NORMALIZATION)
    image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = models['clip_model'].encode_image(image_input)
        # NO NORMALIZATION HERE!
    
    image_embedding = image_features.cpu().numpy()[0]
    
    # Caption
    blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
    caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
    # Text embedding (NO NORMALIZATION)
    text_input = models['tokenizer']([caption]).to(device)
    
    with torch.no_grad():
        text_features = models['clip_model'].encode_text(text_input)
        # NO NORMALIZATION HERE!
    
    text_embedding = text_features.cpu().numpy()[0]
    
    # Combine
    combined_features = np.concatenate([image_embedding, text_embedding])
    
    img_norm = np.linalg.norm(image_embedding)
    text_norm = np.linalg.norm(text_embedding)
    
    return {
        'combined_features': combined_features,
        'caption': caption,
        'img_norm': float(img_norm),
        'text_norm': float(text_norm)
    }

def predict_violence(combined_features, model, device):
    """Predict violence"""
    
    model.eval()
    features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return {
        'prediction': int(np.argmax(probs)),
        'confidence': float(probs[np.argmax(probs)]),
        'violence_prob': float(probs[1]),
        'safe_prob': float(probs[0])
    }

def extract_frames(video_path, fps=1):
    """Extract frames from video"""
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames

# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════

def main():
    
    st.title("🚨 Violence Detection System")
    st.markdown("*Multimodal AI - Vision + Language Fusion*")
    
    st.sidebar.title("⚙️ Settings")
    fps = st.sidebar.slider("Frames per second", 1, 5, 1)
    
    # Test button
    if st.sidebar.button("🧪 Test Model"):
        with st.spinner("Testing..."):
            models = load_models()
            
            violence_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env\combined_features\test\violence"
            
            if os.path.exists(violence_path):
                videos = [v for v in os.listdir(violence_path) if os.path.isdir(os.path.join(violence_path, v))]
                if videos:
                    video_path = os.path.join(violence_path, videos[0])
                    features = [f for f in os.listdir(video_path) if f.endswith('.npy')]
                    if features:
                        feature_path = os.path.join(video_path, features[0])
                        training_feature = np.load(feature_path)
                        
                        result = predict_violence(
                            training_feature,
                            models['violence_model'],
                            models['device']
                        )
                        
                        if result['prediction'] == 1:
                            st.sidebar.success(f"✅ Model works! Violence: {result['confidence']:.1%}")
                        else:
                            st.sidebar.error("❌ Failed")
    
    st.sidebar.info("Expected norms:\n- Image: ~10.5\n- Text: ~7.5")
    
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.video(uploaded_file)
        
        if st.button("🔍 Analyze", type="primary"):
            
            with st.spinner("Loading models..."):
                models = load_models()
            
            with st.spinner(f"Extracting frames..."):
                frames = extract_frames(video_path, fps)
                st.success(f"✅ Extracted {len(frames)} frames")
            
            results = []
            progress = st.progress(0)
            
            for idx, frame in enumerate(frames):
                
                feature_data = extract_features_from_frame(frame, models)
                prediction = predict_violence(
                    feature_data['combined_features'],
                    models['violence_model'],
                    models['device']
                )
                
                results.append({
                    'frame': idx,
                    'timestamp': f"{idx/fps:.1f}s",
                    'prediction': 'Violence' if prediction['prediction'] == 1 else 'Safe',
                    'confidence': prediction['confidence'],
                    'violence_prob': prediction['violence_prob'],
                    'caption': feature_data['caption'],
                    'img_norm': feature_data['img_norm'],
                    'text_norm': feature_data['text_norm'],
                    'frame_image': frame
                })
                
                progress.progress((idx + 1) / len(frames))
            
            violence_count = sum(1 for r in results if r['prediction'] == 'Violence')
            
            st.markdown("---")
            
            if violence_count > 0:
                st.error(f"🚨 **VIOLENCE DETECTED!** {violence_count}/{len(frames)} frames")
            else:
                st.success(f"✅ NO VIOLENCE ({len(frames)} frames)")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames", len(frames))
            col2.metric("Violence", violence_count)
            col3.metric("Safe", len(frames) - violence_count)
            col4.metric("Avg Conf", f"{np.mean([r['confidence'] for r in results]):.1%}")
            
            # Timeline
            st.subheader("📈 Timeline")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            colors = ['red' if r['prediction'] == 'Violence' else 'green' for r in results]
            ax.scatter(range(len(results)), [r['confidence'] for r in results], c=colors, s=100, alpha=0.6)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Confidence')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Details
            st.subheader("🎬 Details")
            
            show_all = st.checkbox("Show all frames", key="show_all")
            
            for i, r in enumerate(results):
                if r['prediction'] == 'Violence' or show_all:
                    with st.expander(f"Frame {i} - {r['timestamp']} - {r['prediction']}"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(r['frame_image'], use_container_width=True)
                        with col2:
                            st.write(f"**Prediction:** {r['prediction']}")
                            st.write(f"**Confidence:** {r['confidence']:.1%}")
                            st.write(f"**Violence:** {r['violence_prob']:.1%}")
                            st.write(f"**Caption:** {r['caption']}")
                            st.write(f"**Img Norm:** {r['img_norm']:.2f} (expect ~10.5)")
                            st.write(f"**Text Norm:** {r['text_norm']:.2f} (expect ~7.5)")
            
            # Export
            df = pd.DataFrame([{
                'Frame': r['frame'],
                'Time': r['timestamp'],
                'Prediction': r['prediction'],
                'Confidence': f"{r['confidence']:.4f}",
                'Violence_Prob': f"{r['violence_prob']:.4f}",
                'Caption': r['caption']
            } for r in results])
            
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
            
            try:
                os.unlink(video_path)
            except:
                pass

if __name__ == "__main__":
    main()
