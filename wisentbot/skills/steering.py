#!/usr/bin/env python3
"""
Wisent Steering Skill - Activation-level behavior control for local models.

Integrates with wisent-ai/wisent for representation engineering:
- Train steering vectors from contrastive pairs
- Apply steering to modify model behavior at inference time
- Detect potentially harmful or hallucinatory outputs
- Self-improve through activation-level corrections
"""

from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillManifest, SkillAction, SkillResult

# Check if wisent is available
HAS_WISENT = False
try:
    from wisent.core.model import Model
    from wisent.core.steering import SteeringMethod, SteeringType
    from wisent.core.contrastive_pairs import ContrastivePairSet, ContrastivePair
    from wisent.core.activations.activations import Activations
    HAS_WISENT = True
except ImportError:
    pass


class SteeringSkill(Skill):
    """
    Skill for activation-level behavior steering using wisent.

    Allows agents running on local models to:
    - Create steering vectors from good/bad example pairs
    - Apply steering to modify behavior without retraining
    - Detect when outputs might be harmful or hallucinatory
    - Self-correct at the activation level
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)

        # Hooks to be set by cognition engine
        self._get_model_fn: Optional[Callable[[], Any]] = None
        self._get_tokenizer_fn: Optional[Callable[[], Any]] = None
        self._is_local_model_fn: Optional[Callable[[], bool]] = None

        # Steering state
        self._steering_method: Optional[Any] = None
        self._contrastive_pairs: List[Dict] = []
        self._steering_vectors: Dict[str, Any] = {}
        self._active_steering: Optional[str] = None
        self._steering_strength: float = 1.0

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
            name="Activation Steering",
            version="1.0.0",
            category="meta",
            description="Control behavior at the activation level using wisent",
            actions=[
                SkillAction(
                    name="add_contrastive_pair",
                    description="Add a good/bad example pair for training steering vectors",
                    parameters={
                        "good_response": {
                            "type": "string",
                            "required": True,
                            "description": "Example of a good/desired response"
                        },
                        "bad_response": {
                            "type": "string",
                            "required": True,
                            "description": "Example of a bad/undesired response"
                        },
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The prompt that generated these responses"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category for this pair (e.g., 'harmful', 'hallucination')"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="train_steering_vector",
                    description="Train a steering vector from collected contrastive pairs",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name for this steering vector"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Only use pairs from this category"
                        },
                        "layer": {
                            "type": "integer",
                            "required": False,
                            "description": "Layer to extract activations from (auto-detected if not set)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="apply_steering",
                    description="Apply a trained steering vector to modify behavior",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the steering vector to apply"
                        },
                        "strength": {
                            "type": "number",
                            "required": False,
                            "description": "Steering strength (default: 1.0, negative to reverse)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="remove_steering",
                    description="Remove active steering and return to base behavior",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_steering_vectors",
                    description="List all trained steering vectors",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="steering_status",
                    description="Get current steering status",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="contrastive_pairs_stats",
                    description="Get statistics about collected contrastive pairs",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="clear_contrastive_pairs",
                    description="Clear all collected contrastive pairs",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Only clear pairs from this category"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="detect_issue",
                    description="Check if a response might be problematic using trained vectors",
                    parameters={
                        "response": {
                            "type": "string",
                            "required": True,
                            "description": "The response to check"
                        },
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The prompt that generated the response"
                        },
                        "vector_name": {
                            "type": "string",
                            "required": False,
                            "description": "Specific vector to check against (checks all if not set)"
                        }
                    },
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
                message="Steering requires local model (vLLM or Transformers), not API"
            )

        if action == "add_contrastive_pair":
            return self._add_contrastive_pair(
                params.get("good_response", ""),
                params.get("bad_response", ""),
                params.get("prompt", ""),
                params.get("category", "general")
            )
        elif action == "train_steering_vector":
            return await self._train_steering_vector(
                params.get("name", ""),
                params.get("category"),
                params.get("layer")
            )
        elif action == "apply_steering":
            return self._apply_steering(
                params.get("name", ""),
                params.get("strength", 1.0)
            )
        elif action == "remove_steering":
            return self._remove_steering()
        elif action == "list_steering_vectors":
            return self._list_steering_vectors()
        elif action == "steering_status":
            return self._steering_status()
        elif action == "contrastive_pairs_stats":
            return self._contrastive_pairs_stats()
        elif action == "clear_contrastive_pairs":
            return self._clear_contrastive_pairs(params.get("category"))
        elif action == "detect_issue":
            return await self._detect_issue(
                params.get("response", ""),
                params.get("prompt", ""),
                params.get("vector_name")
            )
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    def _add_contrastive_pair(
        self,
        good_response: str,
        bad_response: str,
        prompt: str,
        category: str
    ) -> SkillResult:
        """Add a contrastive pair for training."""
        if not good_response.strip() or not bad_response.strip():
            return SkillResult(success=False, message="Both good and bad responses required")
        if not prompt.strip():
            return SkillResult(success=False, message="Prompt required")

        self._contrastive_pairs.append({
            "good": good_response,
            "bad": bad_response,
            "prompt": prompt,
            "category": category,
        })

        return SkillResult(
            success=True,
            message=f"Added contrastive pair ({len(self._contrastive_pairs)} total)",
            data={
                "total_pairs": len(self._contrastive_pairs),
                "category": category,
            }
        )

    async def _train_steering_vector(
        self,
        name: str,
        category: Optional[str],
        layer: Optional[int]
    ) -> SkillResult:
        """Train a steering vector from contrastive pairs."""
        if not name.strip():
            return SkillResult(success=False, message="Vector name required")

        # Filter pairs by category
        pairs = self._contrastive_pairs
        if category:
            pairs = [p for p in pairs if p.get("category") == category]

        if len(pairs) < 3:
            return SkillResult(
                success=False,
                message=f"Need at least 3 contrastive pairs, have {len(pairs)}"
            )

        try:
            model = self._get_model_fn()
            tokenizer = self._get_tokenizer_fn()

            # Create ContrastivePairSet from our pairs
            contrastive_pair_set = ContrastivePairSet()

            for pair in pairs:
                cp = ContrastivePair(
                    prompt=pair["prompt"],
                    positive=pair["good"],
                    negative=pair["bad"],
                )
                contrastive_pair_set.add_pair(cp)

            # Extract activations
            activations = Activations(model=model, tokenizer=tokenizer)
            contrastive_pair_set = activations.extract_contrastive_activations(
                contrastive_pair_set
            )

            # Train steering method
            steering = SteeringMethod(SteeringType.CAA)

            # Auto-detect layer if not specified
            if layer is None:
                # Use middle-to-late layer (typically best for steering)
                num_layers = model.config.num_hidden_layers
                layer = int(num_layers * 0.7)

            training_result = steering.train(contrastive_pair_set, layer_index=layer)

            # Store the trained vector
            self._steering_vectors[name] = {
                "steering": steering,
                "layer": layer,
                "num_pairs": len(pairs),
                "category": category,
                "training_result": training_result,
            }

            return SkillResult(
                success=True,
                message=f"Trained steering vector '{name}' on layer {layer}",
                data={
                    "name": name,
                    "layer": layer,
                    "num_pairs": len(pairs),
                    "category": category,
                }
            )

        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Training failed: {str(e)}"
            )

    def _apply_steering(self, name: str, strength: float) -> SkillResult:
        """Apply a trained steering vector."""
        if name not in self._steering_vectors:
            available = list(self._steering_vectors.keys())
            return SkillResult(
                success=False,
                message=f"Vector '{name}' not found. Available: {available}"
            )

        self._active_steering = name
        self._steering_strength = strength

        vector_info = self._steering_vectors[name]

        return SkillResult(
            success=True,
            message=f"Applied steering '{name}' at strength {strength}",
            data={
                "name": name,
                "strength": strength,
                "layer": vector_info["layer"],
            }
        )

    def _remove_steering(self) -> SkillResult:
        """Remove active steering."""
        if not self._active_steering:
            return SkillResult(
                success=True,
                message="No steering was active"
            )

        old_steering = self._active_steering
        self._active_steering = None
        self._steering_strength = 1.0

        return SkillResult(
            success=True,
            message=f"Removed steering '{old_steering}'",
            data={"removed": old_steering}
        )

    def _list_steering_vectors(self) -> SkillResult:
        """List all trained steering vectors."""
        vectors = {}
        for name, info in self._steering_vectors.items():
            vectors[name] = {
                "layer": info["layer"],
                "num_pairs": info["num_pairs"],
                "category": info["category"],
                "active": name == self._active_steering,
            }

        return SkillResult(
            success=True,
            message=f"Found {len(vectors)} steering vectors",
            data={
                "vectors": vectors,
                "active": self._active_steering,
            }
        )

    def _steering_status(self) -> SkillResult:
        """Get current steering status."""
        return SkillResult(
            success=True,
            message=f"Steering: {'active' if self._active_steering else 'inactive'}",
            data={
                "active_steering": self._active_steering,
                "strength": self._steering_strength if self._active_steering else None,
                "num_vectors": len(self._steering_vectors),
                "num_pairs": len(self._contrastive_pairs),
                "wisent_available": HAS_WISENT,
            }
        )

    def _contrastive_pairs_stats(self) -> SkillResult:
        """Get statistics about contrastive pairs."""
        categories = {}
        for pair in self._contrastive_pairs:
            cat = pair.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

        return SkillResult(
            success=True,
            message=f"Collected {len(self._contrastive_pairs)} contrastive pairs",
            data={
                "total": len(self._contrastive_pairs),
                "by_category": categories,
                "ready_to_train": len(self._contrastive_pairs) >= 3,
            }
        )

    def _clear_contrastive_pairs(self, category: Optional[str]) -> SkillResult:
        """Clear contrastive pairs."""
        if category:
            before = len(self._contrastive_pairs)
            self._contrastive_pairs = [
                p for p in self._contrastive_pairs
                if p.get("category") != category
            ]
            cleared = before - len(self._contrastive_pairs)
            return SkillResult(
                success=True,
                message=f"Cleared {cleared} pairs from category '{category}'",
                data={"cleared": cleared, "remaining": len(self._contrastive_pairs)}
            )
        else:
            cleared = len(self._contrastive_pairs)
            self._contrastive_pairs = []
            return SkillResult(
                success=True,
                message=f"Cleared all {cleared} contrastive pairs",
                data={"cleared": cleared}
            )

    async def _detect_issue(
        self,
        response: str,
        prompt: str,
        vector_name: Optional[str]
    ) -> SkillResult:
        """Detect if a response might be problematic."""
        if not response.strip() or not prompt.strip():
            return SkillResult(success=False, message="Response and prompt required")

        vectors_to_check = []
        if vector_name:
            if vector_name not in self._steering_vectors:
                return SkillResult(
                    success=False,
                    message=f"Vector '{vector_name}' not found"
                )
            vectors_to_check = [(vector_name, self._steering_vectors[vector_name])]
        else:
            vectors_to_check = list(self._steering_vectors.items())

        if not vectors_to_check:
            return SkillResult(
                success=False,
                message="No steering vectors trained yet"
            )

        try:
            model = self._get_model_fn()
            tokenizer = self._get_tokenizer_fn()
            activations = Activations(model=model, tokenizer=tokenizer)

            results = {}
            for name, info in vectors_to_check:
                steering = info["steering"]
                layer = info["layer"]

                # Get activation for this response
                activation = activations.extract_single(
                    prompt=prompt,
                    response=response,
                    layer=layer
                )

                # Check against steering vector
                score = steering.score(activation)
                results[name] = {
                    "score": float(score),
                    "is_problematic": score > 0.5,
                    "category": info["category"],
                }

            any_problematic = any(r["is_problematic"] for r in results.values())

            return SkillResult(
                success=True,
                message=f"Checked against {len(results)} vectors: {'issues detected' if any_problematic else 'no issues'}",
                data={
                    "results": results,
                    "any_problematic": any_problematic,
                }
            )

        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Detection failed: {str(e)}"
            )

    def get_active_steering(self) -> Optional[Dict]:
        """Get active steering info for use in generation."""
        if not self._active_steering:
            return None

        return {
            "name": self._active_steering,
            "strength": self._steering_strength,
            "steering": self._steering_vectors[self._active_steering]["steering"],
            "layer": self._steering_vectors[self._active_steering]["layer"],
        }
