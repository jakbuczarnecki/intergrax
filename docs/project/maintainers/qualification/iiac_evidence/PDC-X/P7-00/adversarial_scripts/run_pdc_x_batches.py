"""Run PDC-X batch tests in ONE bounded pytest process; file list built in Python (no text file)."""
import os, subprocess, sys, time
from pathlib import Path

root = Path.cwd()
files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "tests/qualification/pdc_x").rglob("test_*.py"))
files += [
    "tests/unit/governance/test_pdc_x_inventory.py",
    "tests/unit/governance/test_pdc_x_requirements.py",
    "tests/unit/governance/test_pdc_x_wbs.py",
]
# existing tests modified between the pin and the baseline by PDC-X (non-MCP)
files += [
    "tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_rpl_f_integrity_gates.py",
    "tests/qualification/governance/gr12/test_gr12_a2_mandatory_cla04_composition.py",
    "tests/qualification/governance/strategy/test_gr10_a3_r1_uaep_checkpoint_persistence_resume_qualification.py",
    "tests/qualification/governance/strategy/test_gr10_gates.py",
    "tests/qualification/governance/strategy/test_gr10_r5_inference_inner_governance_qualification.py",
    "tests/qualification/governance/strategy/test_gr10_r6_inference_governance_evidence_qualification.py",
    "tests/unit/docs/test_public_readme_contract.py",
    "tests/unit/rag/vectorstore/test_beta_promotion_readiness.py",
    "tests/unit/rag/vectorstore/test_beta_vector_soak_gate.py",
    "tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py",
]
missing = [f for f in files if not (root / f).is_file()]
print(f"files: {len(files)} missing: {missing}", flush=True)
t = time.time()
rc = subprocess.call([sys.executable, "-m", "pytest", *[f for f in files if f not in missing], "-q", "-rfEx", "--tb=short"])
print(f"exit={rc} elapsed={time.time()-t:.1f}s", flush=True)
sys.exit(rc)
