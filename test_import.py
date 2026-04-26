#!/usr/bin/env python
"""Quick test to verify all imports and basic functionality"""

import sys
sys.path.insert(0, '/app')

print("Testing imports...")

try:
    print("  [PASS] Importing models...")
    from models import (
        EmailTask, AgentAction, StepResult,
        Ticket, HelpdeskAction, HelpdeskEnvState, HelpdeskResetResponse,
        AgentRole, SupportTier, TicketCategory, VALID_ACTION_TYPES
    )
    
    print("  [PASS] Checking HelpdeskResetResponse fields...")
    print(f"    Fields: {HelpdeskResetResponse.model_fields.keys()}")
    
    print("  [PASS] Importing HelpdeskEnv...")
    from helpdeskenv_class import HelpdeskEnv
    
    print("  [PASS] Creating HelpdeskEnv instance...")
    env = HelpdeskEnv()
    
    print("  [PASS] Calling reset()...")
    print(f"    About to reset at line {sys._getframe().f_lineno + 1}")
    reset_response = env.reset()
    print(f"    Reset successful: {type(reset_response)}")
    
except Exception as e:
    print(f"\n[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
