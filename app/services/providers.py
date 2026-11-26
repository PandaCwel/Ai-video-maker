"""Compatibility shim: all inference now runs locally, no hosted APIs."""
from app.services.local_models import LocalModelError as ProviderError
class OpenAICompatibleLLM:
    async def complete(self, system, user):
        import json
        from app.services.local_models import local_llm_json
        return json.dumps(local_llm_json(system,user),ensure_ascii=False)
class ReplicateProvider:
    async def run(self,*args,**kwargs):
        raise ProviderError("Zewnętrzne API usunięte. Użyj lokalnych modeli.")
class ElevenLabsProvider:
    async def clone_voice(self,*args,**kwargs): raise ProviderError("Klonowanie głosu działa lokalnie przez XTTS.")
    async def synthesize(self,*args,**kwargs): raise ProviderError("Synteza działa lokalnie przez XTTS.")
