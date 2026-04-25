#!/usr/bin/env python
"""Quick test to verify all imports and basic functionality"""

import sys
sys.path.insert(0, '/app')

print("Testing imports...")

try:
    print("  ✓ Importing models...")
    from models import (
        EmailTask, Action, AgentAction, StepResult, EnvState, 
        ResetResponse, ErrorResponse, TaskType
    )
    
    print("  ✓ Checking ResetResponse fields...")
    print(f"    Fields: {ResetResponse.model_fields.keys()}")
    
    print("  ✓ Importing emailenv_class...")
    from emailenv_class import EmailEnv
    
    print("  ✓ Creating EmailEnv instance...")
    env = EmailEnv()
    
    print("  ✓ Calling reset()...")
    print(f"    About to reset at line {sys._getframe().f_lineno + 1}")
    reset_response = env.reset()
    print(f"    Reset successful: {type(reset_response)}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!")

