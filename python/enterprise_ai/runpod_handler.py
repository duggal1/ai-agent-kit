import runpod
import torch
from enterprise_ai.trainer import ModelTrainer
from enterprise_ai.model import EnterpriseAIModel

def handler(event):
    try:
        job_type = event["input"]["job_type"]
        
        if job_type == "training":
            # Get training parameters
            model_type = event["input"]["model_type"]
            training_data = event["input"]["training_data"]
            config = event["input"]["config"]
            
            # Initialize trainer
            trainer = ModelTrainer(config)
            
            # Train model
            trainer.train(
                train_data=training_data,
                val_data=training_data,  # Using same data for validation
                epochs=10
            )
            
            # Save model to temporary file
            checkpoint_path = "/tmp/model_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)
            
            return {
                "status": "success",
                "message": "Training completed successfully",
                "model_path": checkpoint_path
            }
            
        elif job_type == "inference":
            model_type = event["input"]["model_type"]
            data = event["input"]["data"]
            
            # Load model
            config = {
                "model_type": model_type,
                # Add other required config parameters
                "num_document_classes": 5,
                "num_entity_types": 10,
                "time_series_features": 3,
                "forecast_horizon": 1
            }
            
            model = EnterpriseAIModel(model_type=model_type, config=config)
            model.eval()
            
            with torch.no_grad():
                outputs = model(data)
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
        return {"status": "error", "error": str(e)}

runpod.serverless.start({"handler": handler})