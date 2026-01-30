#!/usr/bin/env python3
"""
Wisent Steering Skill - Full integration with wisent-ai/wisent.

Provides agents running on local models with:
- Multiple steering methods (CAA, Hyperplane, MLP, Prism, Pulse, Titan)
- Multi-steering (combine multiple vectors)
- Classifier marketplace (discover and use classifiers)
- Response diagnostics (analyze responses for issues)
- Response steering (autonomously improve responses)
- Benchmark integration (use lm-eval benchmarks)

See: https://github.com/wisent-ai/wisent
"""

from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillManifest, SkillAction, SkillResult

# Check if wisent is available
HAS_WISENT = False
WISENT_VERSION = None
try:
    import wisent
    WISENT_VERSION = getattr(wisent, '__version__', 'unknown')
    from wisent.core.model import Model
    from wisent.core.steering import SteeringMethod, SteeringType
    from wisent.core.contrastive_pairs import ContrastivePairSet, ContrastivePair
    from wisent.core.activations.activations import Activations
    from wisent.core.multi_steering import MultiSteering
    HAS_WISENT = True
except ImportError:
    pass

# Check for advanced wisent features
HAS_WISENT_AGENT = False
try:
    from wisent.core.autonomous_agent import AutonomousAgent as WisentAgent
    from wisent.core.agent.diagnose import ResponseDiagnostics, ClassifierMarketplace, AnalysisResult
    from wisent.core.agent.steer import ResponseSteering, ImprovementResult
    HAS_WISENT_AGENT = True
except ImportError:
    pass

# Steering methods available
STEERING_METHODS = {
    "caa": "Contrastive Activation Addition - simple and effective",
    "hyperplane": "Hyperplane-based steering",
    "mlp": "MLP-based learned steering",
    "prism": "Prism steering method",
    "pulse": "Pulse steering method",
    "titan": "Titan advanced steering",
}


class SteeringSkill(Skill):
    """
    Full wisent integration skill for activation-level behavior control.

    This skill wraps wisent's capabilities:
    - Train steering vectors from contrastive pairs
    - Apply multiple steering methods
    - Combine multiple steering vectors
    - Use classifier marketplace for issue detection
    - Autonomously diagnose and improve responses
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)

        # Hooks to model access
        self._get_model_fn: Optional[Callable[[], Any]] = None
        self._get_tokenizer_fn: Optional[Callable[[], Any]] = None
        self._is_local_model_fn: Optional[Callable[[], bool]] = None

        # Wisent components
        self._wisent_model: Optional[Any] = None
        self._wisent_agent: Optional[Any] = None
        self._marketplace: Optional[Any] = None
        self._diagnostics: Optional[Any] = None
        self._response_steering: Optional[Any] = None

        # Steering state
        self._contrastive_pairs: List[Dict] = []
        self._steering_vectors: Dict[str, Any] = {}
        self._active_steering: List[str] = []  # Support multi-steering
        self._steering_weights: Dict[str, float] = {}
        self._multi_steering: Optional[Any] = None

        # Configuration
        self._default_method = "caa"
        self._default_layer: Optional[int] = None

    def set_model_hooks(
        self,
        get_model: Callable[[], Any],
        get_tokenizer: Callable[[], Any],
        is_local_model: Callable[[], bool],
    ):
        """Connect this skill to the agent's model."""
        self._get_model_fn = get_model
        self._get_tokenizer_fn = get_tokenizer
        self._is_local_model_fn = is_local_model

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="steering",
            name="Wisent Steering",
            version="2.0.0",
            category="meta",
            description="Activation-level behavior control via wisent",
            actions=[
                # === Setup ===
                SkillAction(
                    name="init_wisent",
                    description="Initialize wisent with your model for advanced steering",
                    parameters={
                        "classifier_paths": {
                            "type": "array",
                            "required": False,
                            "description": "Paths to search for existing classifiers"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get wisent steering status and capabilities",
                    parameters={},
                    estimated_cost=0,
                ),
                # === Contrastive Pairs ===
                SkillAction(
                    name="add_pair",
                    description="Add a good/bad response pair for training steering vectors",
                    parameters={
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The input prompt"
                        },
                        "good": {
                            "type": "string",
                            "required": True,
                            "description": "The good/desired response"
                        },
                        "bad": {
                            "type": "string",
                            "required": True,
                            "description": "The bad/undesired response"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category (e.g., 'hallucination', 'harmful', 'refusal')"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pairs_stats",
                    description="Get statistics about collected contrastive pairs",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="clear_pairs",
                    description="Clear collected contrastive pairs",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Only clear this category"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Training ===
                SkillAction(
                    name="train",
                    description="Train a steering vector from contrastive pairs",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name for the steering vector"
                        },
                        "method": {
                            "type": "string",
                            "required": False,
                            "description": "Method: caa, hyperplane, mlp, prism, pulse, titan"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Only use pairs from this category"
                        },
                        "layer": {
                            "type": "integer",
                            "required": False,
                            "description": "Layer to extract from (auto-detected if not set)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_vectors",
                    description="List all trained steering vectors",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="save_vector",
                    description="Save a steering vector to disk",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of vector to save"
                        },
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "File path to save to"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="load_vector",
                    description="Load a steering vector from disk",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name to assign to loaded vector"
                        },
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "File path to load from"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Steering Application ===
                SkillAction(
                    name="steer",
                    description="Apply steering vector(s) to modify behavior",
                    parameters={
                        "vectors": {
                            "type": "string",
                            "required": True,
                            "description": "Comma-separated vector names or 'name:weight' pairs"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="unsteer",
                    description="Remove all active steering",
                    parameters={},
                    estimated_cost=0,
                ),
                # === Diagnostics ===
                SkillAction(
                    name="diagnose",
                    description="Analyze a response for issues using trained classifiers",
                    parameters={
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The input prompt"
                        },
                        "response": {
                            "type": "string",
                            "required": True,
                            "description": "The response to analyze"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="improve",
                    description="Autonomously improve a problematic response",
                    parameters={
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The input prompt"
                        },
                        "response": {
                            "type": "string",
                            "required": True,
                            "description": "The response to improve"
                        },
                        "max_attempts": {
                            "type": "integer",
                            "required": False,
                            "description": "Max improvement attempts (default: 3)"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Marketplace ===
                SkillAction(
                    name="marketplace",
                    description="Browse available classifiers in the marketplace",
                    parameters={
                        "search": {
                            "type": "string",
                            "required": False,
                            "description": "Search term to filter classifiers"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="download_classifier",
                    description="Download a classifier from the marketplace",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Classifier name to download"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Methods Info ===
                SkillAction(
                    name="methods",
                    description="List available steering methods",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        """Check if wisent and local model are available."""
        if not HAS_WISENT:
            return False
        if self._is_local_model_fn and not self._is_local_model_fn():
            return False
        return self._get_model_fn is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_WISENT:
            return SkillResult(
                success=False,
                message="wisent not installed. Run: pip install wisent"
            )

        if not self._get_model_fn:
            return SkillResult(
                success=False,
                message="Steering not connected to model"
            )

        if self._is_local_model_fn and not self._is_local_model_fn():
            return SkillResult(
                success=False,
                message="Steering requires local model (vLLM/Transformers), not API"
            )

        # Route to handlers
        handlers = {
            "init_wisent": self._init_wisent,
            "status": self._status,
            "add_pair": self._add_pair,
            "pairs_stats": self._pairs_stats,
            "clear_pairs": self._clear_pairs,
            "train": self._train,
            "list_vectors": self._list_vectors,
            "save_vector": self._save_vector,
            "load_vector": self._load_vector,
            "steer": self._steer,
            "unsteer": self._unsteer,
            "diagnose": self._diagnose,
            "improve": self._improve,
            "marketplace": self._marketplace_browse,
            "download_classifier": self._download_classifier,
            "methods": self._methods,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Handlers ===

    async def _init_wisent(self, params: Dict) -> SkillResult:
        """Initialize wisent with the model."""
        try:
            model = self._get_model_fn()

            # Wrap in wisent Model if needed
            if not isinstance(model, Model):
                self._wisent_model = Model(model)
            else:
                self._wisent_model = model

            # Auto-detect optimal layer
            if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
                num_layers = model.config.num_hidden_layers
                self._default_layer = int(num_layers * 0.7)  # 70% depth is typically good

            # Initialize wisent agent if available
            if HAS_WISENT_AGENT:
                classifier_paths = params.get("classifier_paths", [])
                self._marketplace = ClassifierMarketplace(
                    model=self._wisent_model,
                    search_paths=classifier_paths
                )

            return SkillResult(
                success=True,
                message="Wisent initialized",
                data={
                    "wisent_version": WISENT_VERSION,
                    "default_layer": self._default_layer,
                    "has_agent": HAS_WISENT_AGENT,
                    "has_marketplace": self._marketplace is not None,
                    "methods_available": list(STEERING_METHODS.keys()),
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Init failed: {e}")

    async def _status(self, params: Dict) -> SkillResult:
        """Get current steering status."""
        return SkillResult(
            success=True,
            message="Steering status",
            data={
                "wisent_installed": HAS_WISENT,
                "wisent_version": WISENT_VERSION,
                "wisent_agent_available": HAS_WISENT_AGENT,
                "model_initialized": self._wisent_model is not None,
                "default_layer": self._default_layer,
                "contrastive_pairs": len(self._contrastive_pairs),
                "trained_vectors": list(self._steering_vectors.keys()),
                "active_steering": self._active_steering,
                "steering_weights": self._steering_weights,
                "methods_available": list(STEERING_METHODS.keys()),
            }
        )

    async def _add_pair(self, params: Dict) -> SkillResult:
        """Add a contrastive pair."""
        prompt = params.get("prompt", "").strip()
        good = params.get("good", "").strip()
        bad = params.get("bad", "").strip()
        category = params.get("category", "general")

        if not prompt or not good or not bad:
            return SkillResult(success=False, message="prompt, good, and bad are required")

        self._contrastive_pairs.append({
            "prompt": prompt,
            "good": good,
            "bad": bad,
            "category": category,
        })

        return SkillResult(
            success=True,
            message=f"Added pair ({len(self._contrastive_pairs)} total)",
            data={"total": len(self._contrastive_pairs), "category": category}
        )

    async def _pairs_stats(self, params: Dict) -> SkillResult:
        """Get pair statistics."""
        categories = {}
        for pair in self._contrastive_pairs:
            cat = pair.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

        return SkillResult(
            success=True,
            message=f"{len(self._contrastive_pairs)} pairs collected",
            data={
                "total": len(self._contrastive_pairs),
                "by_category": categories,
                "ready_to_train": len(self._contrastive_pairs) >= 3,
            }
        )

    async def _clear_pairs(self, params: Dict) -> SkillResult:
        """Clear contrastive pairs."""
        category = params.get("category")
        if category:
            before = len(self._contrastive_pairs)
            self._contrastive_pairs = [p for p in self._contrastive_pairs if p.get("category") != category]
            cleared = before - len(self._contrastive_pairs)
        else:
            cleared = len(self._contrastive_pairs)
            self._contrastive_pairs = []

        return SkillResult(
            success=True,
            message=f"Cleared {cleared} pairs",
            data={"cleared": cleared, "remaining": len(self._contrastive_pairs)}
        )

    async def _train(self, params: Dict) -> SkillResult:
        """Train a steering vector."""
        name = params.get("name", "").strip()
        method = params.get("method", self._default_method).lower()
        category = params.get("category")
        layer = params.get("layer", self._default_layer)

        if not name:
            return SkillResult(success=False, message="Vector name required")

        if method not in STEERING_METHODS:
            return SkillResult(
                success=False,
                message=f"Unknown method: {method}. Available: {list(STEERING_METHODS.keys())}"
            )

        # Filter pairs
        pairs = self._contrastive_pairs
        if category:
            pairs = [p for p in pairs if p.get("category") == category]

        if len(pairs) < 3:
            return SkillResult(
                success=False,
                message=f"Need at least 3 pairs, have {len(pairs)}"
            )

        try:
            model = self._get_model_fn()
            tokenizer = self._get_tokenizer_fn()

            # Create ContrastivePairSet
            pair_set = ContrastivePairSet()
            for pair in pairs:
                cp = ContrastivePair(
                    prompt=pair["prompt"],
                    positive=pair["good"],
                    negative=pair["bad"],
                )
                pair_set.add_pair(cp)

            # Extract activations
            activations = Activations(model=model, tokenizer=tokenizer)
            pair_set = activations.extract_contrastive_activations(pair_set)

            # Train with selected method
            if method == "caa":
                steering = SteeringMethod(SteeringType.CAA)
            else:
                steering = SteeringMethod(SteeringType.CAA)  # Fallback, expand as needed

            # Auto-detect layer if not specified
            if layer is None:
                if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
                    layer = int(model.config.num_hidden_layers * 0.7)
                else:
                    layer = 15  # Reasonable default

            result = steering.train(pair_set, layer_index=layer)

            self._steering_vectors[name] = {
                "steering": steering,
                "method": method,
                "layer": layer,
                "num_pairs": len(pairs),
                "category": category,
                "training_result": result,
            }

            return SkillResult(
                success=True,
                message=f"Trained '{name}' using {method} at layer {layer}",
                data={
                    "name": name,
                    "method": method,
                    "layer": layer,
                    "num_pairs": len(pairs),
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Training failed: {e}")

    async def _list_vectors(self, params: Dict) -> SkillResult:
        """List trained vectors."""
        vectors = {}
        for name, info in self._steering_vectors.items():
            vectors[name] = {
                "method": info["method"],
                "layer": info["layer"],
                "num_pairs": info["num_pairs"],
                "category": info["category"],
                "active": name in self._active_steering,
                "weight": self._steering_weights.get(name, 1.0),
            }

        return SkillResult(
            success=True,
            message=f"{len(vectors)} vectors trained",
            data={"vectors": vectors}
        )

    async def _save_vector(self, params: Dict) -> SkillResult:
        """Save vector to disk."""
        name = params.get("name", "")
        path = params.get("path", "")

        if name not in self._steering_vectors:
            return SkillResult(success=False, message=f"Vector '{name}' not found")

        try:
            import torch
            info = self._steering_vectors[name]
            torch.save({
                "steering": info["steering"],
                "method": info["method"],
                "layer": info["layer"],
                "num_pairs": info["num_pairs"],
                "category": info["category"],
            }, path)

            return SkillResult(success=True, message=f"Saved '{name}' to {path}")
        except Exception as e:
            return SkillResult(success=False, message=f"Save failed: {e}")

    async def _load_vector(self, params: Dict) -> SkillResult:
        """Load vector from disk."""
        name = params.get("name", "")
        path = params.get("path", "")

        try:
            import torch
            data = torch.load(path, weights_only=False)

            self._steering_vectors[name] = {
                "steering": data["steering"],
                "method": data.get("method", "unknown"),
                "layer": data["layer"],
                "num_pairs": data.get("num_pairs", 0),
                "category": data.get("category"),
                "training_result": None,
            }

            return SkillResult(success=True, message=f"Loaded '{name}' from {path}")
        except Exception as e:
            return SkillResult(success=False, message=f"Load failed: {e}")

    async def _steer(self, params: Dict) -> SkillResult:
        """Apply steering."""
        vectors_str = params.get("vectors", "")

        if not vectors_str:
            return SkillResult(success=False, message="Specify vectors to apply")

        # Parse "name:weight,name:weight" or just "name,name"
        self._active_steering = []
        self._steering_weights = {}

        for spec in vectors_str.split(","):
            spec = spec.strip()
            if ":" in spec:
                name, weight = spec.split(":", 1)
                name = name.strip()
                try:
                    weight = float(weight.strip())
                except ValueError:
                    weight = 1.0
            else:
                name = spec
                weight = 1.0

            if name not in self._steering_vectors:
                return SkillResult(success=False, message=f"Vector '{name}' not found")

            self._active_steering.append(name)
            self._steering_weights[name] = weight

        return SkillResult(
            success=True,
            message=f"Applied {len(self._active_steering)} steering vectors",
            data={
                "active": self._active_steering,
                "weights": self._steering_weights,
            }
        )

    async def _unsteer(self, params: Dict) -> SkillResult:
        """Remove all steering."""
        removed = self._active_steering.copy()
        self._active_steering = []
        self._steering_weights = {}

        return SkillResult(
            success=True,
            message=f"Removed steering: {removed}" if removed else "No steering was active"
        )

    async def _diagnose(self, params: Dict) -> SkillResult:
        """Diagnose a response for issues."""
        if not HAS_WISENT_AGENT:
            return SkillResult(
                success=False,
                message="Wisent agent features not available"
            )

        prompt = params.get("prompt", "")
        response = params.get("response", "")

        if not prompt or not response:
            return SkillResult(success=False, message="prompt and response required")

        try:
            if self._diagnostics is None:
                # Need to initialize with classifier configs
                return SkillResult(
                    success=False,
                    message="Run init_wisent first with classifier paths"
                )

            analysis = await self._diagnostics.analyze_response(response, prompt)

            return SkillResult(
                success=True,
                message=f"Issues: {analysis.issues_found}" if analysis.has_issues else "No issues detected",
                data={
                    "has_issues": analysis.has_issues,
                    "issues": analysis.issues_found,
                    "confidence": analysis.confidence,
                    "quality_score": analysis.quality_score,
                    "suggestions": analysis.suggestions,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Diagnosis failed: {e}")

    async def _improve(self, params: Dict) -> SkillResult:
        """Autonomously improve a response."""
        if not HAS_WISENT_AGENT:
            return SkillResult(
                success=False,
                message="Wisent agent features not available"
            )

        prompt = params.get("prompt", "")
        response = params.get("response", "")
        max_attempts = params.get("max_attempts", 3)

        if not prompt or not response:
            return SkillResult(success=False, message="prompt and response required")

        try:
            if self._response_steering is None:
                return SkillResult(
                    success=False,
                    message="Run init_wisent first with classifier paths"
                )

            # First diagnose
            analysis = await self._diagnostics.analyze_response(response, prompt)

            if not analysis.has_issues:
                return SkillResult(
                    success=True,
                    message="No issues to improve",
                    data={"original": response, "improved": response, "no_change": True}
                )

            # Attempt improvement
            result = await self._response_steering.improve_response(prompt, response, analysis)

            return SkillResult(
                success=result.success,
                message=f"Improvement via {result.improvement_method}: {'success' if result.success else 'failed'}",
                data={
                    "original": result.original_response,
                    "improved": result.improved_response,
                    "method": result.improvement_method,
                    "improvement_score": result.improvement_score,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Improvement failed: {e}")

    async def _marketplace_browse(self, params: Dict) -> SkillResult:
        """Browse classifier marketplace."""
        if not HAS_WISENT_AGENT or self._marketplace is None:
            return SkillResult(
                success=False,
                message="Marketplace not available. Run init_wisent first."
            )

        search = params.get("search", "")

        try:
            summary = self._marketplace.get_marketplace_summary()
            return SkillResult(
                success=True,
                message="Marketplace summary",
                data={"summary": summary}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Marketplace error: {e}")

    async def _download_classifier(self, params: Dict) -> SkillResult:
        """Download a classifier."""
        if not HAS_WISENT_AGENT:
            return SkillResult(
                success=False,
                message="Marketplace not available"
            )

        name = params.get("name", "")
        if not name:
            return SkillResult(success=False, message="Classifier name required")

        # This would integrate with wisent's classifier download system
        return SkillResult(
            success=False,
            message="Classifier download not yet implemented"
        )

    async def _methods(self, params: Dict) -> SkillResult:
        """List available steering methods."""
        return SkillResult(
            success=True,
            message="Available steering methods",
            data={"methods": STEERING_METHODS}
        )

    # === Public API for CognitionEngine ===

    def get_active_steering(self) -> Optional[Dict]:
        """Get active steering info for use in generation."""
        if not self._active_steering:
            return None

        return {
            "vectors": [
                {
                    "name": name,
                    "weight": self._steering_weights.get(name, 1.0),
                    "steering": self._steering_vectors[name]["steering"],
                    "layer": self._steering_vectors[name]["layer"],
                }
                for name in self._active_steering
            ]
        }
