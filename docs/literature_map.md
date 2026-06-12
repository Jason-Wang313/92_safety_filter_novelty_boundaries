        # Literature Map

        Paper: 92 safety_filter_novelty_boundaries

        Field box: safe robot learning

        Thesis: Safety Filter Novelty Boundaries turns the seed bet into a mechanism: Identify when safety filters change the learned mechanism rather than merely clipping actions.

        ## Landscape Sweep Summary
        The selector ranked records from the shared 500,000-record pool. The closest visible clusters are:
        - Safe robot arm with safe joint mechanism using nonlinear spring system for collision safety (2009)
- Safe joint mechanism using inclined link with springs for collision safety and positioning accuracy of a robot arm (2010)
- Safe joint mechanism based on nonlinear stiffness for safe human-robot collision (2008)
- Safe Link Mechanism based on Passive Compliance for Safe Human-Robot Collision (2007)
- A Nonlinear Stiffness Safe Joint Mechanism Design for Human Robot Interaction (2010)
- Safe robust multi-agent reinforcement learning with neural control barrier functions and safety attention mechanism (2025)
- Safe prescribed time controller for wheeled mobile robots by using control barrier functions as a safety filter (2025)
- Towards a Snake-Like Flexible Robot With Variable Stiffness Using an SMA Spring-Based Friction Change Mechanism (2022)
- Did We Get Sensorimotor Adaptation Wrong? Implicit Adaptation as Direct Policy Updating Rather than Forward-Model-Based Learning (2021)
- Embodying rather than encoding: Towards developing a source-filter theory for undulation gait generation (2024)
- Design and Control of a Discrete Variable Stiffness Actuator With Instant Stiffness Switch for Safe Human-Robot Interaction (2021)
- Safety-Critical Whole-Body Control for Humanoid Robots via Input-to-State Safe Control Barrier Functions (2026)

        ## Hidden Assumptions
        - The executed trajectory is a sufficient training target.
- Unobserved physical alternatives can be averaged into uncertainty.
- Task labels capture the mechanism that caused failure.
- A planner only needs nominal feasibility.
- Embodiment-specific contact effects are nuisance variation.

        ## Boundary
        The project avoids weak moves such as bigger models, generic uncertainty, or a benchmark-only contribution. It centers a mechanism-level change: Safety filter novelty boundaries keeps action-critical alternatives explicit until a physical observation collapses them.
