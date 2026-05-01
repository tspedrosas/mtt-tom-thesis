# Each dataset key maps to a list of human-defined, mutually-exclusive principles.
PRINCIPLES = {
    "strategy_qa": [
        ["1", "Temporal reasoning (before/after dates)", "Confuses timelines or state changes (before/after, still/already, lifespan)."],
        ["2", "Negation trap (question contains 'not')", "Misinterprets negation or its scope (not/never/no, double negatives)."],
        ["3", "Numeric superlatives (largest, smallest, longest)", "Misreads extrema or totals across sets/time (largest/most/fewest/longest)."],
        ["4", "Entity nationality conflation", "Confuses origin/nationality with location/brand/citizenship (e.g., founder vs company)."],
        ["5", "False premise acceptance", "Treats a false presupposition as true instead of rejecting the premise."],
        ["6", "Fiction - Reality confusion (super-hero, mythical, cartoon)", "Applies fictional-world facts to reality (or vice versa)."],
        ["7", "Category mismatch / commonsense (biology vs product, entity roles)", "Chooses an answer from the wrong semantic category or role; violates basic commonsense constraints."],
        ["8", "Physical plausibility / order-of-magnitude error", "Violates basic physical limits or scale plausibility (speed/height/weight/distance)."]
    ],
    "ec_qa": [
        ["1", "Type/Role Constraint Violation", "Chooses an answer of the wrong category/role (e.g., picks an object when the question seeks a place, or an event when a container is needed)"],
        ["2", "Typicality / Prototypicality & Scale Bias", "Picks a common but not most likely option given context or scale (e.g., drawer vs university for “tens of thousands of paper clips”)"],
        ["3", "World-Knowledge Granularity / Mapping Error", "Confuses levels like continent/country/place or institution roles (e.g., church vs Europe)"],
        ["4", "Physical Affordances & Spatial/Containment Error", "Misunderstands where things are stored/used or how objects interact "],
        ["5", "Causal/Teleological Misalignment", "Reverses cause/effect or confuses purpose with outcome"],
        ["6", "Quantifier/Comparative Misinterpretation", "For items truly involving quantities, comparatives, or superlatives"]
    ],
    "gsm8k": [
        ["1", "Step sequencing / operation ordering error", "Executes multi-step arithmetic in the wrong order, drops a required step, or mixes add/sub/mul/div across steps."],
        ["2", "Mis-copied intermediate or final transcription", "Carries forward the wrong number from the scratchpad or outputs an intermediate instead of the final result (incl. place-value typos)."],
        ["3", "Equation / relation set-up error", "Translates the verbal relations into the wrong equation (rate×time, parts/wholes, systems of relations, etc.)."],
        ["4", "Percent/ratio composition error", "Misapplies percentages/ratios over multiple steps (e.g., successive discounts, 'of the remainder' vs 'of the original', compounding)."],
        ["5", "Unit/scale conversion slip", "Confuses units or scales (MB<->GB, hours<->minutes, cents<->dollars; per-unit vs total)."],    
        ["6", "Off-by-one / boundary handling", "Inclusive/exclusive count mistakes, fencepost errors, or unwarranted rounding."]
    ]
}
