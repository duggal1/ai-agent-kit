import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'DocumentAnalyzer',
    'EntityRecognizer',
    'SemanticAnalyzer',
    'DocumentClassifier',
    'FinancialClassifier',
    'HybridClassifier'
]

class DocumentAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get('embedding_dim', 1024)
        
        # Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ) for _ in range(6)  # Increased depth
        ])
        
        # Specialized financial feature extraction
        self.financial_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Document structure analysis
        self.structure_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )
        
        # Final feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x):
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply self-attention
        attn_output, _ = self.self_attention(x, x, x)
        
        # Financial analysis
        financial_features = self.financial_analyzer(attn_output)
        
        # Structure analysis
        structure_features = self.structure_analyzer(attn_output)
        
        # Combine features
        combined = torch.cat([
            attn_output, 
            financial_features,
            structure_features
        ], dim=-1)
        
        # Final feature extraction with global pooling
        pooled = combined.mean(dim=1)
        output = self.feature_extractor(pooled)
        
        return output

class EntityRecognizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get('embedding_dim', 1024)
        
        # Simplified architecture with forced output size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)  # Force output size
        )
    
    def forward(self, x):
        # Force mean pooling and ensure output size
        pooled = torch.mean(x, dim=1)  # [batch_size, hidden_size]
        return self.projection(pooled)  # Guaranteed [batch_size, hidden_size]

class SemanticAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get('embedding_dim', 1024)
        
        # Simplified architecture with forced output size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)  # Force output size
        )
    
    def forward(self, x):
        # Force mean pooling and ensure output size
        pooled = torch.mean(x, dim=1)  # [batch_size, hidden_size]
        return self.projection(pooled)  # Guaranteed [batch_size, hidden_size]

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