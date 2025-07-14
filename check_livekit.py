import livekit.agents
import pkgutil

# This will list submodules of livekit.agents
print("Submodules of livekit.agents:")
for importer, modname, ispkg in pkgutil.iter_modules(livekit.agents.__path__):
    print(f"- {modname} (package: {ispkg})")

# If 'llm' exists, let's dive into it
try:
    import livekit.agents.llm
    print("\nSubmodules of livekit.agents.llm:")
    for importer, modname, ispkg in pkgutil.iter_modules(livekit.agents.llm.__path__):
        print(f"- {modname} (package: {ispkg})")
except ImportError:
    print("\n'livekit.agents.llm' not found or not a package with submodules.")