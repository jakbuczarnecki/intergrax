# © Artur Czarnecki. All rights reserved.

"""Cross-model behavioral comparison over qualification outcomes (DS-E2E-15J-L2)."""

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.analyzers import (
    CohortExecutionBehaviorAnalyzer,
    QualificationExitBehaviorAnalyzer,
    SessionLifecycleBehaviorAnalyzer,
    default_behavior_analyzers,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    ANALYSIS_TASK_ID,
    AreaAnalysisStatus,
    BehavioralComparisonResult,
    ComparisonArea,
    ComparisonAreaFinding,
    CrossModelAnalysisStatus,
    CrossModelBehavioralAnalysisRequest,
    ModelIdentityRef,
    OutcomeSourceRef,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.engine import (
    run_cross_model_behavioral_analysis,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.protocol import (
    BehaviorAnalyzer,
)

__all__ = [
    "ANALYSIS_TASK_ID",
    "AreaAnalysisStatus",
    "BehaviorAnalyzer",
    "BehavioralComparisonResult",
    "CohortExecutionBehaviorAnalyzer",
    "ComparisonArea",
    "ComparisonAreaFinding",
    "CrossModelAnalysisStatus",
    "CrossModelBehavioralAnalysisRequest",
    "ModelIdentityRef",
    "OutcomeSourceRef",
    "QualificationExitBehaviorAnalyzer",
    "SessionLifecycleBehaviorAnalyzer",
    "default_behavior_analyzers",
    "run_cross_model_behavioral_analysis",
]
