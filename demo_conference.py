#!/usr/bin/env python3
"""
Conference Demo Script - Find Best Images for Visualization
Automatically find the best images from each dataset for conference presentation
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import random
from collections import defaultdict

# Add gradcam scripts to path
sys.path.append(str(Path(__file__).parent / "scripts" / "gradcam"))
from improved_gradcam import safe_grad_cam, find_best_layers_for_gradcam


def find_best_images_for_datasets():
    """Find best images from each dataset for conference demo"""
    
    print("🔍 Finding best images for conference visualization...")
    print("=" * 60)
    
    # Dataset configurations with available models
    dataset_configs = {
        'RAF-DB': {
            'data_path': 'data/dataset/rafdb_basic/test',
            'models': [
                {
                    'path': 'scripts/gradcam/model_weights/VGG_CBAM_RAFDB.h5.keras',
                    'type': 'vgg16',
                    'attention': 'CBAM',
                    'name': 'VGG16 + CBAM'
                },
                {
                    'path': 'scripts/gradcam/model_weights/VGG_SCSE_RAFDB.h5.keras', 
                    'type': 'vgg16',
                    'attention': 'scSE',
                    'name': 'VGG16 + scSE'
                }
            ],
            'input_size': (224, 224),
            'emotion_labels': ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        },
        'FER-2013': {
            'data_path': 'data/dataset/FER-2013/test',
            'models': [
                # Add FER-2013 models here when available
                # {
                #     'path': 'path/to/VGG_CBAM_FER2013.h5',
                #     'type': 'vgg16', 
                #     'attention': 'CBAM',
                #     'name': 'VGG16 + CBAM (FER-2013)'
                # }
            ],
            'input_size': (96, 96),
            'emotion_labels': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        }
    }
    
    best_images = {}
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n📊 Processing {dataset_name} dataset...")
        
        if not config['models']:
            print(f"⚠️  No models available for {dataset_name}, skipping...")
            continue
            
        dataset_path = Path(config['data_path'])
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            continue
            
        # Find test images
        image_files = []
        for emotion_dir in dataset_path.iterdir():
            if emotion_dir.is_dir():
                emotion_images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
                image_files.extend([(img, emotion_dir.name) for img in emotion_images])
        
        if not image_files:
            print(f"❌ No images found in {dataset_path}")
            continue
            
        print(f"📁 Found {len(image_files)} images")
        
        # Sample subset for testing (to avoid processing all images)
        max_test_images = min(50, len(image_files))
        sampled_images = random.sample(image_files, max_test_images)
        
        # Load first available model for evaluation
        model_config = config['models'][0]
        try:
            model = load_model(model_config['path'], compile=False)
            print(f"✅ Loaded model: {model_config['name']}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
            
        # Evaluate images to find best ones
        image_scores = []
        
        for i, (img_path, true_emotion) in enumerate(sampled_images):
            if i % 10 == 0:
                print(f"  📈 Evaluating... {i+1}/{len(sampled_images)}")
                
            try:
                # Load and preprocess image
                img = image.load_img(img_path, target_size=config['input_size'], color_mode='rgb')
                img_array = image.img_to_array(img) / 255.0
                
                # Get prediction
                pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
                confidence = np.max(pred)
                predicted_class = np.argmax(pred)
                predicted_emotion = config['emotion_labels'][predicted_class]
                
                # Calculate score (higher is better)
                # Prioritize: high confidence + correct prediction + good for visualization
                correct_prediction = (predicted_emotion.lower() == true_emotion.lower())
                score = confidence * (2.0 if correct_prediction else 1.0)
                
                image_scores.append({
                    'path': img_path,
                    'true_emotion': true_emotion,
                    'predicted_emotion': predicted_emotion,
                    'confidence': confidence,
                    'correct': correct_prediction,
                    'score': score,
                    'img_array': img_array
                })
                
            except Exception as e:
                print(f"  ⚠️  Error processing {img_path.name}: {e}")
                continue
        
        # Sort by score and select best images
        image_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top images for each emotion
        emotion_best = defaultdict(list)
        for img_data in image_scores:
            emotion = img_data['true_emotion']
            if len(emotion_best[emotion]) < 2:  # Top 2 per emotion
                emotion_best[emotion].append(img_data)
        
        # Select overall best images
        top_images = image_scores[:5]  # Top 5 overall
        
        best_images[dataset_name] = {
            'top_overall': top_images,
            'by_emotion': dict(emotion_best),
            'model_configs': config['models'],
            'input_size': config['input_size'],
            'emotion_labels': config['emotion_labels']
        }
        
        # Print results
        print(f"\n🏆 Best images for {dataset_name}:")
        for i, img_data in enumerate(top_images, 1):
            print(f"  {i}. {img_data['path'].name}")
            print(f"     True: {img_data['true_emotion']}, Pred: {img_data['predicted_emotion']}")
            print(f"     Confidence: {img_data['confidence']:.3f}, Score: {img_data['score']:.3f}")
            print(f"     Correct: {'✅' if img_data['correct'] else '❌'}")
    
    return best_images


def create_conference_demo_with_best_images(best_images):
    """Create focused demo for conference presentation using best images"""
    
    print("\n🎯 Creating Conference Demo with Best Images")
    print("=" * 50)
    
    if not best_images:
        print("❌ No best images found")
        return
    
    # Create output directory
    output_dir = Path("conference_best_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, dataset_data in best_images.items():
        print(f"\n📊 Creating demo for {dataset_name}...")
        
        # Use the best image
        best_image_data = dataset_data['top_overall'][0]
        img_array = best_image_data['img_array']
        img_path = best_image_data['path']
        
        print(f"🖼️  Using best image: {img_path.name}")
        print(f"   True: {best_image_data['true_emotion']}, Pred: {best_image_data['predicted_emotion']}")
        print(f"   Confidence: {best_image_data['confidence']:.3f}")
        
        # Process all available models for this dataset
        all_results = {}
        
        for model_config in dataset_data['model_configs']:
            print(f"\n🔄 Processing {model_config['name']}...")
            
            try:
                # Load model
                model = load_model(model_config['path'], compile=False)
                print(f"✅ Loaded {model_config['name']}")
                
                # Get prediction
                pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
                predicted_emotion = dataset_data['emotion_labels'][np.argmax(pred)]
                confidence = np.max(pred)
                print(f"🎯 Prediction: {predicted_emotion} ({confidence:.3f})")
                
                # Select key layers for demonstration
                if model_config['type'] == 'vgg16':
                    demo_layers = {
                        'Early (Block1)': 'Conv1.2',
                        'Mid (Block3)': 'Conv3.2', 
                        'Late (Block5)': 'Conv5.2'
                    }
                elif model_config['type'] == 'resnet18':
                    demo_layers = {
                        'Early (Conv1)': 'Conv1',
                        'Mid (Stage2)': 'Conv3_x2_Block1_2_elu',
                        'Late (Stage4)': 'Conv5_x2_Block1_2_elu'
                    }
                else:
                    demo_layers = {}
                
                # Generate GradCAMs
                model_heatmaps = {}
                for stage_name, layer_name in demo_layers.items():
                    heatmap = safe_grad_cam(model, img_array, layer_name, dataset_data['emotion_labels'])
                    if heatmap is not None:
                        model_heatmaps[stage_name] = heatmap
                        print(f"  ✅ {stage_name}: intensity {np.mean(heatmap):.3f}")
                    else:
                        print(f"  ❌ {stage_name}: Failed")
                
                all_results[model_config['name']] = {
                    'heatmaps': model_heatmaps,
                    'prediction': predicted_emotion,
                    'confidence': confidence
                }
                
            except Exception as e:
                print(f"  ❌ Error with {model_config['name']}: {e}")
        
        # Create visualization for this dataset
        if all_results:
            create_attention_comparison_figure(
                img_array, all_results, output_dir, 
                img_path, dataset_name, f"{dataset_name}_best"
            )
            create_intensity_progression_figure(
                all_results, output_dir, img_path, f"{dataset_name}_best"
            )
    
    print(f"\n🎉 Conference demo completed!")
    print(f"📁 Results saved in: {output_dir}")
    
    # List generated files
    print(f"\n📋 Generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")


def create_attention_comparison_figure(img_array, all_results, output_dir, demo_image, dataset_type, prefix=""):
    """Create main comparison figure for conference"""
    
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models + 1, 4, figsize=(16, 4 * (n_models + 1)))
    
    # Handle single model case
    if n_models == 1:
        axes = axes.reshape(n_models + 1, -1)
    
    fig.suptitle('Attention Mechanism Comparison on Facial Emotion Recognition', 
                 fontsize=16, fontweight='bold')
    
    # Top row: Original image and legend
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title(f'Original Image\n({dataset_type} Dataset)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Legend for stages
    stage_names = ['Early Features\n(Low-level)', 'Mid Features\n(Pattern)', 'Late Features\n(Semantic)']
    for i, stage_name in enumerate(stage_names):
        if i + 1 < axes.shape[1]:
            axes[0, i + 1].text(0.5, 0.5, stage_name, ha='center', va='center',
                               transform=axes[0, i + 1].transAxes, fontsize=12, fontweight='bold')
            axes[0, i + 1].axis('off')
    
    # Model rows
    for row, (model_name, results) in enumerate(all_results.items(), 1):
        heatmaps = results['heatmaps']
        prediction = results['prediction']
        confidence = results['confidence']
        
        # Model info
        axes[row, 0].text(0.5, 0.7, model_name, ha='center', va='center',
                         transform=axes[row, 0].transAxes, fontsize=14, fontweight='bold')
        axes[row, 0].text(0.5, 0.3, f'Pred: {prediction}\nConf: {confidence:.3f}', 
                         ha='center', va='center', transform=axes[row, 0].transAxes, fontsize=10)
        axes[row, 0].axis('off')
        
        # Attention heatmaps
        stage_list = list(heatmaps.keys())[:3]  # Take first 3 stages
        for col, stage_name in enumerate(stage_list, 1):
            heatmap = heatmaps[stage_name]
            
            # Resize heatmap to match image
            heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            
            # Create overlay
            heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.6 * img_array + 0.4 * heatmap_colored
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'{stage_name}\nIntensity: {np.mean(heatmap):.3f}', 
                                   fontweight='bold', fontsize=10)
            axes[row, col].axis('off')
        
        # Fill remaining columns
        for col in range(len(stage_list) + 1, axes.shape[1]):
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    img_name = Path(demo_image).stem if hasattr(demo_image, 'stem') else Path(str(demo_image)).stem
    output_name = f"{prefix}_{img_name}_attention_comparison.png" if prefix else f"{img_name}_attention_comparison.png"
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  💾 Saved: {output_path}")
    
    plt.close(fig)


def create_intensity_progression_figure(all_results, output_dir, demo_image, prefix=""):
    """Create intensity progression comparison"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    model_names = list(all_results.keys())
    stage_names = ['Early', 'Mid', 'Late']
    
    # Calculate intensities
    model_intensities = {}
    for model_name, results in all_results.items():
        heatmaps = results['heatmaps']
        intensities = []
        
        stage_list = list(heatmaps.keys())[:3]
        for stage in stage_list:
            if stage in heatmaps:
                intensities.append(np.mean(heatmaps[stage]))
            else:
                intensities.append(0)
        
        model_intensities[model_name] = intensities
    
    # Plot
    x = np.arange(len(stage_names))
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, intensities) in enumerate(model_intensities.items()):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, intensities, width, 
                     label=model_name, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels
        for bar, intensity in zip(bars, intensities):
            if intensity > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{intensity:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Network Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Intensity', fontsize=12, fontweight='bold')
    ax.set_title('Attention Intensity Progression Across Network Stages', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    ax.text(0.02, 0.98, 
           'Higher intensity = Stronger attention focus\nEarly: Edge/texture detection\nMid: Pattern recognition\nLate: Semantic understanding',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    img_name = Path(demo_image).stem if hasattr(demo_image, 'stem') else Path(str(demo_image)).stem
    output_name = f"{prefix}_{img_name}_intensity_progression.png" if prefix else f"{img_name}_intensity_progression.png"
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  💾 Saved: {output_path}")
    
    plt.close(fig)


def check_prerequisites():
    """Check if all required files exist"""
    print("🔍 Checking prerequisites...")
    
    # Check models
    model_dir = Path("scripts/gradcam/model_weights")
    required_models = [
        'VGG_CBAM_RAFDB.h5.keras',
        'VGG_SCSE_RAFDB.h5.keras'
    ]
    
    missing_models = []
    for model in required_models:
        if not (model_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ Missing models: {missing_models}")
        return False
    
    # Check dataset
    dataset_dir = Path("data/dataset/FER-2013/test")
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return False
    
    print("✅ All prerequisites found")
    return True


def main():
    """Main demo function - Find best images and create conference visualization"""
    print("🎯 Conference Demo - Best Image Finder")
    print("=" * 40)
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Step 1: Find best images from each dataset
        print("Step 1: Finding best images from datasets...")
        best_images = find_best_images_for_datasets()
        
        if not best_images:
            print("❌ No best images found. Please check dataset paths and models.")
            return
            
        # Step 2: Create conference demo with best images
        print("\nStep 2: Creating conference visualization...")
        create_conference_demo_with_best_images(best_images)
        
        print("\n🎉 Conference demo pipeline completed successfully!")
        print("📋 Summary:")
        for dataset_name, data in best_images.items():
            if data['top_overall']:
                best_img = data['top_overall'][0]
                print(f"  - {dataset_name}: {best_img['path'].name} (confidence: {best_img['confidence']:.3f})")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()