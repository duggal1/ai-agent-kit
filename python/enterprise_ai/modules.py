import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# Define and export DOMAIN_PATTERNS
DOMAIN_PATTERNS: Dict[str, List[str]] = {
    'technical': ['api', 'endpoint', 'service', 'database', 'cloud', 'server', 'latency', 'performance'],
    'financial': ['revenue', 'cost', 'profit', 'loss', 'margin', 'budget', 'forecast', 'growth'],
    'security': ['vulnerability', 'threat', 'encryption', 'firewall', 'breach', 'incident', 'risk'],
    'metadata': ['version', 'timestamp', 'author', 'classification', 'status', 'category'],
    'legal': ['compliance', 'regulation', 'policy', 'contract', 'agreement', 'terms', 'liability'],
    'medical': ['diagnosis', 'treatment', 'patient', 'clinical', 'medical', 'healthcare', 'prescription']
}

__all__ = [
    'DocumentClassifier',
    'FinancialClassifier',
    'HybridClassifier',
    'DocumentAnalyzer',
    'EntityRecognizer',
    'SemanticAnalyzer',
    'DomainAdapter',
    'MultiScaleFeatureFusion',
    'ConfidenceCalibration',
    'DOMAIN_PATTERNS'  # Add DOMAIN_PATTERNS to __all__
]

class DocumentAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get('embedding_dim', 1024)
                         
        # Deeper transformer stack with gradient checkpointing
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=32,  # Increased attention heads
                dim_feedforward=hidden_size * 8,  # Wider network
                dropout=0.15,
                batch_first=True,
                activation='gelu'
            ) for _ in range(12)  # Doubled depth
        ])
        
        # Advanced financial analysis module
        self.financial_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            ResidualBlock(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Enhanced document structure analysis
        self.structure_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            ResidualBlock(hidden_size * 2),
            SelfAttentionBlock(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Multi-scale attention mechanism
        self.attention_scales = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=2**i,  # Variable heads per scale
                dropout=0.1,
                batch_first=True
            ) for i in range(3, 6)  # 8, 16, 32 heads
        ])
        
        # Advanced feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 6),
            nn.LayerNorm(hidden_size * 6),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_size * 6),
            SelfAttentionBlock(hidden_size * 6),
            nn.Linear(hidden_size * 6, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.GELU(),
            nn.Linear(hidden_size * 3, hidden_size)
        )
        
        # Confidence boosting module
        self.confidence_booster = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x):
        # Ensure input is 2D [batch_size, hidden_size]
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Average over sequence length
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Process through analyzers
        financial_features = self.financial_analyzer(x)
        structure_features = self.structure_analyzer(x)
        
        # Multi-scale attention
        x_3d = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        attention_outputs = []
        for attention in self.attention_scales:
            attn_out, _ = attention(x_3d, x_3d, x_3d)
            attention_outputs.append(attn_out.squeeze(1))
        
        # Combine features
        combined = torch.cat([
            financial_features,
            structure_features,
            *attention_outputs
        ], dim=-1)
        
        # Final processing
        output = self.feature_fusion(combined)
        output = self.confidence_booster(output)
        
        return output

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = x.unsqueeze(1) if x.dim() == 2 else x
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        out = self.norm(residual + attn_out.squeeze(1))
        return out

class EntityRecognizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get('embedding_dim', 1024)
        
        # Enhanced entity recognition
        self.entity_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            ResidualBlock(hidden_size * 2),
            SelfAttentionBlock(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Entity confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
    
    def forward(self, x):
        # Process entities with attention
        entity_features = self.entity_processor(x)
        
        # Boost confidence
        confidence_scores = self.confidence_scorer(entity_features)
        output = entity_features * torch.sigmoid(confidence_scores)
        
        return output

class SemanticAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get('embedding_dim', 1024)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Add dimension handling layers
        self.input_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Ensure input is 3D [batch_size, seq_len, hidden_size]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            
        # Normalize input
        x = self.input_norm(x)
        
        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        
        # Project features
        projected = self.projection(attn_output)
        
        # Normalize and mean pool
        output = self.output_norm(projected)
        return output.mean(dim=1)  # Return [batch_size, hidden_size]

class DocumentClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class FinancialClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.financial_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.financial_attention(x, x, x)
        return self.classifier(attn_out.squeeze(1))

class HybridClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        return self.classifier(attn_out.squeeze(1))

class DomainAdapter(nn.Module):
    def __init__(self, hidden_size: int, domain: str):
        super().__init__()
        self.domain = domain
        
        # Verify domain exists
        if domain not in DOMAIN_PATTERNS:
            raise ValueError(f"Invalid domain: {domain}. Available domains: {list(DOMAIN_PATTERNS.keys())}")
        
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            ResidualBlock(hidden_size * 4),
            SelfAttentionBlock(hidden_size * 4, num_heads=16),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Domain-specific patterns
        self.pattern_recognizer = PatternRecognizer(
            hidden_size=hidden_size,
            patterns=DOMAIN_PATTERNS[domain]
        )

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_heads: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        
        # Adjust projection dimensions
        self.input_projection = nn.Linear(input_size, output_size)
        self.input_norm = nn.LayerNorm(output_size)
        
        # Feature projection with corrected dimensions
        self.projection = nn.Sequential(
            nn.Linear(output_size, output_size * 2),
            nn.LayerNorm(output_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_size * 2, output_size)
        )
        
        # Add attention layer with correct dimensions
        self.attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, x):
        # Ensure input is 2D [batch_size, hidden_size]
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Project input to correct dimension
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add sequence dimension for attention
        x_3d = x.unsqueeze(1)  # [batch_size, 1, output_size]
        
        # Apply self-attention
        attn_out, _ = self.attention(x_3d, x_3d, x_3d)
        
        # Remove sequence dimension
        x = attn_out.squeeze(1)
        
        # Final projection
        output = self.projection(x)
        
        return output  # [batch_size, output_size]

class EnhancedEntityRecognizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_types = {
            'technical': TechnicalEntityDetector(),
            'financial': FinancialEntityDetector(),
            'security': SecurityEntityDetector(),
            'metadata': MetadataEntityDetector()
        }
        
        self.context_processor = ContextualProcessor(
            hidden_size=config['hidden_size'],
            num_layers=6
        )

class PatternRecognizer(nn.Module):
    def __init__(self, hidden_size: int, patterns: List[str]):
        super().__init__()
        self.patterns = patterns
        self.pattern_embeddings = nn.Parameter(
            torch.randn(len(patterns), hidden_size)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        pattern_scores, _ = self.attention(
            x,
            self.pattern_embeddings.unsqueeze(0).expand(x.size(0), -1, -1),
            self.pattern_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)
        )
        return pattern_scores.mean(dim=1)

class ScaleProcessor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.scale_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # Scale-specific processing
        B, L, D = x.shape
        scaled_len = L // self.scale_factor
        scaled_x = F.adaptive_avg_pool2d(
            x.transpose(1, 2).unsqueeze(-1),
            (D, scaled_len)
        ).squeeze(-1).transpose(1, 2)
        
        attn_out, _ = self.attention(scaled_x, scaled_x, scaled_x)
        scaled_out = self.norm(attn_out + scaled_x)
        return self.scale_proj(scaled_out)

class CrossScaleAttention(nn.Module):
    def __init__(self, hidden_size: int, num_scales: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fusion = nn.Linear(hidden_size * num_scales, hidden_size)
        
    def forward(self, scale_outputs: List[torch.Tensor]):
        # Cross-scale attention
        cross_attn_outputs = []
        for i, scale_out in enumerate(scale_outputs):
            others = torch.cat([s for j, s in enumerate(scale_outputs) if j != i], dim=1)
            attn_out, _ = self.cross_attention(scale_out, others, others)
            cross_attn_outputs.append(attn_out)
        
        # Fusion of all scales
        combined = torch.cat(cross_attn_outputs, dim=-1)
        return self.fusion(combined)

class EntityDetector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn_out, _ = self.entity_attention(x, x, x)
        return self.entity_classifier(attn_out)

# Specialized entity detectors
class TechnicalEntityDetector(EntityDetector):
    def __init__(self):
        super().__init__(hidden_size=1024)

class FinancialEntityDetector(EntityDetector):
    def __init__(self):
        super().__init__(hidden_size=1024)

class SecurityEntityDetector(EntityDetector):
    def __init__(self):
        super().__init__(hidden_size=1024)

class MetadataEntityDetector(EntityDetector):
    def __init__(self):
        super().__init__(hidden_size=1024)

class ContextualProcessor(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.context_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.context_layers:
            x = layer(x)
        return x

class PatternMatcher(nn.Module):
    def __init__(self, hidden_size: int, num_patterns: int, pattern_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patterns = num_patterns
        self.pattern_dim = pattern_dim
        
        # Learnable patterns
        self.patterns = nn.Parameter(
            torch.randn(num_patterns, pattern_dim)
        )
        
        # Pattern matching network
        self.matcher = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, num_patterns)
        )
        
        # Pattern attention
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=pattern_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get pattern matching scores
        pattern_scores = self.matcher(x)
        
        # Apply attention over patterns
        pattern_context = self.patterns.unsqueeze(0).expand(x.size(0), -1, -1)
        attn_output, _ = self.pattern_attention(x, pattern_context, pattern_context)
        
        # Combine pattern scores with attention
        combined_scores = F.softmax(pattern_scores, dim=-1) * torch.sigmoid(
            torch.bmm(attn_output, self.patterns.transpose(0, 1))
        )
        
        return combined_scores

class ConfidenceCalibration(nn.Module):
    """Module for calibrating confidence scores of model predictions."""
    def __init__(self, input_size: int = None, num_bins: int = 15, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.num_bins = num_bins
        
        # Handle both hidden_size and input_size for backward compatibility
        self.input_size = input_size
        
        if input_size is not None:
            # Calibration network
            self.calibration_network = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.LayerNorm(input_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_size // 2, num_bins),
                nn.Softmax(dim=-1)
            )
        
        # Bin boundaries
        self.bin_boundaries = nn.Parameter(
            torch.linspace(0, 1, num_bins + 1),
            requires_grad=False
        )
        
        # Learnable scaling factors for each bin
        self.bin_scaling = nn.Parameter(torch.ones(num_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply confidence calibration
        Args:
            x: Input tensor, either logits or hidden states
        Returns:
            Calibrated confidence scores
        """
        # Temperature scaling
        if self.input_size is not None:
            # Process high-dimensional input
            bin_weights = self.calibration_network(x)
            base_confidence = torch.sigmoid(x.mean(dim=-1))
        else:
            # Process logits directly
            scaled_logits = x / self.temperature
            bin_weights = F.softmax(scaled_logits, dim=-1)
            base_confidence = bin_weights.max(dim=-1)[0]
        
        # Apply bin-wise calibration
        calibrated_confidence = torch.zeros_like(base_confidence)
        for i in range(self.num_bins):
            bin_mask = (base_confidence >= self.bin_boundaries[i]) & (base_confidence < self.bin_boundaries[i + 1])
            calibrated_confidence[bin_mask] = base_confidence[bin_mask] * self.bin_scaling[i]
        
        # Combine with bin weights if using calibration network
        if self.input_size is not None:
            final_confidence = (calibrated_confidence.unsqueeze(-1) * bin_weights).sum(dim=-1)
        else:
            final_confidence = calibrated_confidence
        
        return torch.clamp(final_confidence, 0, 1)