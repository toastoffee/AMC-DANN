import torch
from torch import nn
from grl import GradientReversalFunction
from mcldnn import MCLDNNFeatureExtractor, MCLDNNClassifier

class DANN(nn.Module):
    def __init__(self, grl_alpha=1.0):
        super(DANN, self).__init__()

        self.feature_extractor = MCLDNNFeatureExtractor()

        self.classifier = MCLDNNClassifier(num_classes=11)

        self.domain_classifier = MCLDNNClassifier(num_classes=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, alpha: float):
        feature = self.feature_extractor(x)

        reversed_features = GradientReversalFunction.apply(feature, alpha)

        # class classification
        class_logits = self.classifier(feature)
        # domain classification
        domain_logits = self.domain_classifier(reversed_features)

        return class_logits, domain_logits


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = DANN()

    sgn, _ = net(sgn, 1.0)

    print(sgn.shape)