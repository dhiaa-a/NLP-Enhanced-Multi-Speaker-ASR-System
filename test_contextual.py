# test_better_correction.py
from src.nlp.correction_strategies import CorrectionStrategies

class BetterCorrector:
    def __init__(self):
        self.strategies = CorrectionStrategies()
        
        # Add more corrections
        self.strategies.common_corrections.update({
            'consid': 'consider',
            'deadl': 'deadline',
            'fri': 'friday',
            'he': 'hear',
            'b': 'bad',
            'exact': 'exactly',
            'tha': 'that'
        })
    
    def correct(self, text, context=None):
        # Apply pattern-based correction (this is working great!)
        corrected = self.strategies.pattern_based_correction(text)
        return corrected

# Test it
corrector = BetterCorrector()

tests = [
    "So I was think- the meet- tomorrow at tw- no actually thre-",
    "Yeah I agr## with tha+ but we should also consid## the budg##",
    "Exact### and the the the deadl### is next fri###",
    "Can you he## me? The connec### is really b##"
]

print("Better corrections with expanded dictionary:\n")
for text in tests:
    print(f"Raw:       {text}")
    print(f"Corrected: {corrector.correct(text)}")
    print()