import runpod
import torch
from enterprise_ai.trainer import AdvancedModelTrainer
from enterprise_ai.model import EnterpriseAIModel
import os
import logging

logger = logging.getLogger(__name__)

def handler(event):
    try:
        job_type = event["input"]["job_type"]
        
        if job_type == "training":
            logger.info("Starting training job")
            model_type = event["input"]["model_type"]
            training_data = event["input"]["training_data"]
            config = event["input"]["config"]
            
            # Initialize trainer with GPU support
            trainer = AdvancedModelTrainer(config)
            
            # Convert training data to proper format
            processed_data = trainer._prepare_data(training_data)
            
            # Train model
            trainer.train(
                train_loader=processed_data['train_loader'],
                val_loader=processed_data['val_loader'],
                epochs=config.get('epochs', 10)
            )
            
            # Save model to shared storage
            model_path = f"/runpod-volume/models/{model_type}_{event['id']}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            trainer.model.save(model_path)
            
            return {
                "status": "success",
                "message": "Training completed successfully",
                "model_path": model_path
            }
            
        elif job_type == "inference":
            logger.info("Starting inference job")
            model_type = event["input"]["model_type"]
            data = event["input"]["data"]
            
            # Load model from shared storage
            model_path = f"/runpod-volume/models/{model_type}_latest.pt"
            model = EnterpriseAIModel.load(model_path)
            model.eval()
            
            # Move model to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            with torch.no_grad():
                # Prepare input data
                processed_data = model._prepare_inference_data(data)
                processed_data = {k: v.to(device) for k, v in processed_data.items()}
                
                # Run inference
                outputs = model(processed_data)
                
                # Process outputs
                prediction = {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs.items()
                }
                
            return {
                "status": "success",
                "prediction": prediction
            }
            
        else:
            raise ValueError(f"Unknown job type: {job_type}")
            
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

# Initialize RunPod
runpod.serverless.start({
    "handler": handler,
    "refresh_worker": True,
    "enable_gpu": True
})