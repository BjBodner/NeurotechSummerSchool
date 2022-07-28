from relaxation_score_calculator import RelaxationScoreCalculator
from print_logger import PrintLogger

logger = PrintLogger()
relaxation_score_calculator = RelaxationScoreCalculator()

while True:

    relaxation_score, feature_vector = relaxation_score_calculator.calc_score()
    logger.log(relaxation_score, feature_vector)
