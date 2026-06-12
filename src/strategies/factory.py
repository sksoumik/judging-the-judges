"""Factory for creating strategy instances from config."""

from src.judges.base import JudgeBase
from src.judges.factory import JudgeFactory
from src.strategies.base import StrategyBase
from src.strategies.calibrated_rubric import CalibratedRubricStrategy
from src.strategies.combined import CombinedBudgetStrategy, CombinedFullStrategy
from src.strategies.cot_forcing import CoTForcingStrategy
from src.strategies.ensemble_cross import EnsembleCrossStrategy
from src.strategies.ensemble_same import EnsembleSameStrategy
from src.strategies.naive import NaiveStrategy
from src.strategies.position_swap import PositionSwapStrategy
from src.strategies.reference_guided import ReferenceGuidedStrategy


class StrategyFactory:
    """Creates strategy instances from configuration."""

    def __init__(self, judge_factory: JudgeFactory):
        self.judge_factory = judge_factory

    def create(
        self,
        strategy_name: str,
        primary_model: str,
        ensemble_models: list[str] | None = None,
    ) -> StrategyBase:
        judge = self.judge_factory.create(primary_model)

        match strategy_name:
            case "B0_naive":
                return NaiveStrategy(judge)
            case "S1_position_swap":
                return PositionSwapStrategy(judge)
            case "S2_ensemble_same":
                return EnsembleSameStrategy(judge)
            case "S3_ensemble_cross":
                models = ensemble_models or ["gemini-2.5-pro", "claude-sonnet-4", "gemini-2.5-flash"]
                # Exclude primary model if it's in the list, replace with primary
                judges = [self.judge_factory.create(m) for m in models]
                return EnsembleCrossStrategy(judges)
            case "S4_calibrated_rubric":
                return CalibratedRubricStrategy(judge)
            case "S5_cot_forcing":
                return CoTForcingStrategy(judge)
            case "S6_reference_guided":
                return ReferenceGuidedStrategy(judge)
            case "S7_combined_full":
                models = ensemble_models or ["gemini-2.5-pro", "claude-sonnet-4", "gemini-2.5-flash"]
                judges = [self.judge_factory.create(m) for m in models]
                return CombinedFullStrategy(judges)
            case "S8_combined_budget":
                return CombinedBudgetStrategy(judge)
            case _:
                raise ValueError(f"Unknown strategy: {strategy_name}")
