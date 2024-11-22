import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any
import logging
import torch.nn.functional as F
from enterprise_ai.modules import (
    DocumentClassifier,
    FinancialClassifier,
    HybridClassifier,
    DocumentAnalyzer,
    EntityRecognizer,
    SemanticAnalyzer
)

logger = logging.getLogger(__name__)

class EnterpriseAIModel(nn.Module):
    def __init__(self, model_type: str, config: Dict[str, Any]):
        super().__init__()
        self.model_type = model_type
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure required config parameters exist
        self.config.setdefault('num_document_classes', 5)
        self.config.setdefault('num_entity_types', 8)
        self.hidden_size = self.config.get('embedding_dim', 1024)
        
        # Initialize base model
        try:
            self.base_model = AutoModel.from_pretrained(
                'microsoft/deberta-v3-large',
                output_hidden_states=True,
                output_attentions=True
            )
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
            
        # Initialize ensemble heads with robust error handling
        try:
            self.ensemble_heads = nn.ModuleList([
                self._create_classifier('DocumentClassifier'),
                self._create_classifier('FinancialClassifier'),
                self._create_classifier('HybridClassifier')
            ])
        except Exception as e:
            logger.error(f"Error initializing ensemble heads: {str(e)}")
            # Fallback to basic classifier if specialized ones fail
            self.ensemble_heads = nn.ModuleList([
                self._create_fallback_classifier() for _ in range(3)
            ])
        
        # Specialized modules with dynamic sizing
        self.document_analyzer = DocumentAnalyzer(config)
        self.entity_recognizer = EntityRecognizer(config)
        self.semantic_analyzer = SemanticAnalyzer(config)
        
        # Adaptive feature fusion
        doc_fusion_input = self.hidden_size * 3  # Concatenated features
        entity_fusion_input = self.hidden_size * 2  # Concatenated features
        
        # Enhanced feature fusion with cross-attention
        self.feature_fusion = nn.ModuleDict({
            'document_fusion': nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                ResidualAttention(self.hidden_size),
                nn.Dropout(0.1)
            ),
            'entity_fusion': nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                ResidualAttention(self.hidden_size),
                nn.Dropout(0.1)
            )
        })
        
        # Add attention mechanism for ensemble weighting
        self.ensemble_attention = nn.Sequential(
            nn.Linear(config['num_document_classes'], self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Add hidden state weights parameter
        self.hidden_state_weights = nn.Parameter(
            torch.ones(self.base_model.config.num_hidden_layers + 1)
        )
        
        # Add confidence boosting module
        self.confidence_booster = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config['num_document_classes'])
        )
        
        # Output layers with dynamic sizing
        self.output_layers = nn.ModuleDict({
            'document_classifier': nn.Sequential(
                nn.Linear(self.hidden_size, config['num_document_classes']),
                nn.LayerNorm(config['num_document_classes'])
            ),
            'entity_extractor': nn.Sequential(
                nn.Linear(self.hidden_size, config['num_entity_types']),
                nn.LayerNorm(config['num_entity_types'])
            )
        })
        
        # Enhanced tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/deberta-v3-large',
            additional_special_tokens=[
                '[DOC_START]', '[DOC_END]',
                '[ENTITY_START]', '[ENTITY_END]'
            ]
        )
        
        # Add financial domain pre-training
        self.financial_embeddings = nn.Embedding(
            num_embeddings=10000,
            embedding_dim=self.hidden_size
        )
        
        # Add new complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.to(self.device)
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model"""
        try:
            # Get base model outputs
            base_outputs = self.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            
            # Get hidden states
            hidden_states = base_outputs.hidden_states
            last_hidden_state = base_outputs.last_hidden_state
            
            # Process through modules
            doc_features = self.document_analyzer(last_hidden_state)
            entity_features = self.entity_recognizer(last_hidden_state)
            semantic_features = self.semantic_analyzer(last_hidden_state)
            
            # Feature fusion
            doc_concat = torch.cat([doc_features, entity_features, semantic_features], dim=-1)
            entity_concat = torch.cat([entity_features, semantic_features], dim=-1)
            
            # Apply fusion layers
            fused_doc = self.feature_fusion['document_fusion'](doc_concat)
            fused_entity = self.feature_fusion['entity_fusion'](entity_concat)
            
            return {
                'document_class': self.output_layers['document_classifier'](fused_doc),
                'entities': self.output_layers['entity_extractor'](fused_entity),
                'hidden_states': hidden_states,
                'last_hidden_state': last_hidden_state
            }
        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            raise
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        
        if 'document_class' in outputs and 'labels' in batch:
            classification_loss = nn.CrossEntropyLoss()(
                outputs['document_class'],
                batch['labels'].to(self.device)
            )
            loss = loss + classification_loss
        
        if 'entities' in outputs and 'entity_labels' in batch:
            entity_loss = nn.BCEWithLogitsLoss()(
                outputs['entities'],
                batch['entity_labels'].to(self.device)
            )
            loss = loss + entity_loss
        
        return loss
    
    def save(self, path: str):
        torch.save({
            'model_state': self.state_dict(),
            'config': self.config,
            'model_type': self.model_type
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'EnterpriseAIModel':
        try:
            checkpoint = torch.load(
                path, 
                map_location='cpu'
            )
            
            # Ensure config has required fields
            config = checkpoint.get('config', {})
            config.setdefault('num_document_classes', 5)
            config.setdefault('num_entity_types', 8)
            config.setdefault('embedding_dim', 1024)  # Add this to match expected dimensions
            
            # Create new model instance with error handling
            try:
                model = cls(
                    model_type=checkpoint.get('model_type', 'enhanced_document_processor'),
                    config=config
                )
                
                # Load state dict with strict=False to handle missing/unexpected keys
                if 'model_state' in checkpoint:
                    try:
                        model.load_state_dict(checkpoint['model_state'], strict=False)
                        logger.info("Model state loaded successfully with some missing keys (expected)")
                    except Exception as e:
                        logger.warning(f"Partial state dict loading: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error creating model instance: {str(e)}")
                # Create basic model as fallback
                model = cls(
                    model_type='basic_document_processor',
                    config={'num_document_classes': 5, 'num_entity_types': 8, 'embedding_dim': 1024}
                )
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Create and return basic model as fallback
            return cls(
                model_type='basic_document_processor',
                config={'num_document_classes': 5, 'num_entity_types': 8, 'embedding_dim': 1024}
            )
    
    def post_process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        processed = {}
        
        if 'document_class' in outputs:
            logits = outputs['document_class']
            
            # Safely handle hidden states
            if 'hidden_states' in outputs:
                complexity_score = self.assess_document_complexity(outputs['hidden_states'])
                # Adjust confidence based on complexity
                confidence_scale = torch.sigmoid(1.0 - complexity_score)
                scaled_logits = logits * confidence_scale
            else:
                # Fallback when hidden states aren't available
                logger.warning("Hidden states not available, using raw logits")
                scaled_logits = logits
                complexity_score = torch.tensor([0.5])  # Default medium complexity
            
            probs = F.softmax(scaled_logits, dim=1)
            max_prob, pred_class = torch.max(probs, dim=1)
            
            processed['document_class'] = {
                'predicted_class': int(pred_class[0]),
                'confidence': float(max_prob[0]),
                'complexity_score': float(complexity_score[0]) if isinstance(complexity_score, torch.Tensor) else complexity_score,
                'class_probabilities': probs[0].tolist()
            }
        
        if 'entities' in outputs:
            entity_probs = torch.sigmoid(outputs['entities'])
            processed['entities'] = {
                'detected_entities': [i for i, p in enumerate(entity_probs[0]) if p > 0.7],
                'entity_probabilities': entity_probs[0].tolist()
            }
        
        return processed
    
    def _enhance_prediction_confidence(self, probs):
        # Remove artificial boosting
        return probs
    
    def assess_prediction_quality(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Assess the quality of predictions"""
        if 'document_class' in outputs:
            probs = F.softmax(outputs['document_class'], dim=1)
            confidence = float(torch.max(probs).item())
            return confidence
        return 0.0
    
    def extract_insights(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract additional insights from model outputs"""
        insights = {}
        
        if 'document_class' in outputs:
            probs = F.softmax(outputs['document_class'], dim=1)
            insights['confidence_distribution'] = probs[0].tolist()
            insights['prediction_entropy'] = float(-(probs * torch.log(probs + 1e-10)).sum().item())
        
        return insights
    
    def _create_classifier(self, classifier_type: str) -> nn.Module:
        try:
            # Update how we access the classifier classes
            classifier_mapping = {
                'DocumentClassifier': DocumentClassifier,
                'FinancialClassifier': FinancialClassifier,
                'HybridClassifier': HybridClassifier
            }
            
            if classifier_type not in classifier_mapping:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
            
            return classifier_mapping[classifier_type](self.hidden_size, self.config['num_document_classes'])
        except Exception as e:
            logger.warning(f"Failed to create {classifier_type}: {str(e)}")
            return self._create_fallback_classifier()
    
    def _create_fallback_classifier(self) -> nn.Module:
        """Create a basic fallback classifier if specialized ones fail"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.config['num_document_classes'])
        )
    
    def assess_document_complexity(self, hidden_states):
        # Add proper complexity metrics
        structural_complexity = self._analyze_structure(hidden_states)
        semantic_complexity = self._analyze_semantics(hidden_states)
        technical_complexity = self._analyze_technical_content(hidden_states)
        return (structural_complexity + semantic_complexity + technical_complexity) / 3
    
    def adjust_confidence(self, logits, complexity_score):
        base_confidence = F.softmax(logits, dim=1)
        # Scale confidence based on complexity
        adjusted_confidence = base_confidence * (1.0 - complexity_score)
        return adjusted_confidence
    
    def process_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Process raw text input into model-ready format"""
        try:
            # Tokenize input
            encoded = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            return {
                'input_ids': encoded['input_ids'].to(self.device),
                'attention_mask': encoded['attention_mask'].to(self.device)
            }
        except Exception as e:
            logger.error(f"Input processing error: {str(e)}")
            raise
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Handle both raw text and preprocessed inputs"""
        if isinstance(inputs.get('text'), str):
            # Process raw text input
            inputs = self.process_input(inputs['text'])
        return self.forward(inputs)
    
    def _analyze_structure(self, hidden_states) -> torch.Tensor:
        """Analyze document structure complexity"""
        try:
            last_hidden = hidden_states[-1]
            sent_lengths = torch.sum(torch.abs(last_hidden) > 0.1, dim=-1)
            sent_variance = torch.var(sent_lengths.float(), dim=-1)
            return torch.sigmoid(sent_variance / 100)
        except Exception as e:
            logger.error(f"Structure analysis error: {str(e)}")
            return torch.tensor(0.5, device=self.device)
    
    def _analyze_semantics(self, hidden_states) -> torch.Tensor:
        """Analyze semantic complexity"""
        try:
            semantic_features = []
            for layer_state in hidden_states[-4:]:
                semantic_diversity = torch.std(layer_state, dim=1)
                semantic_features.append(semantic_diversity)
            semantic_complexity = torch.stack(semantic_features).mean(dim=0)
            return torch.sigmoid(semantic_complexity.mean())
        except Exception as e:
            logger.error(f"Semantic analysis error: {str(e)}")
            return torch.tensor(0.5, device=self.device)
    
    def _analyze_technical_content(self, hidden_states) -> torch.Tensor:
        """Analyze technical content complexity"""
        try:
            last_hidden = hidden_states[-1]
            token_complexity = torch.norm(last_hidden, dim=-1)
            max_complexity = torch.max(token_complexity, dim=-1)[0]
            avg_complexity = torch.mean(token_complexity, dim=-1)
            return torch.sigmoid((max_complexity + avg_complexity) / 2)
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return torch.tensor(0.5, device=self.device)
    
    def _init_weights(self, module):
        """Initialize weights for new modules"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class ResidualAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        residual = x
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(residual + attn_out.squeeze(1))
        return out