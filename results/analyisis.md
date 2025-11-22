python src/experiment/analyze_nl4opt_pot_failures.py 
[load_nl4opt_numeric] Loading from hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json
[load_nl4opt_numeric] Using 231 numeric examples.

=== Global stats (PoT vs Strong PoT) ===
Total common examples: 231
  fixed (baseline wrong -> strong correct): 58
  regressed (baseline correct -> strong wrong): 22
  both wrong: 25

[fixed] Showing up to 3 examples...
================================================================================
[FIXED] Example index = 10
================================================================================
Question:
  A magic school sends letters to student either by carrier pigeons or owls. A carrier pigeon can carry two letters at a time and requires 3 treats for service. An owl can carry 5 letters at a time and requires 5 treats for service.  At most 40% of the birds can be owls. In addition, the school only has 1000 treats available and at least 20 carrier pigeons must be uses. How many of each bird can be used to maximize the total number of letters that can be sent.

Ground truth objective: 841.0

Baseline PoT:
  parsed_answer = 842.1052631578947
  correct       = False
  abs_error     = 1.105263157894683

  Baseline PoT code snippet:
    def solve():
        # We model:
        # x = number of carrier pigeons
        # y = number of owls
        #
        # Maximize letters:  L = 2x + 5y
        #
        # Constraints:
        # 1) Treats: 3x + 5y <= 1000
        # 2) Owl fraction: y <= 0.4(x + y)  ->  0.6y <= 0.4x  ->  3y <= 2x  ->  y <= (2/3)x
        # 3) At least 20 carrier pigeons: x >= 20
        # 4) x, y >= 0
        #
        # This is a small 2-variable LP; we can solve it analytically by checking corner points.

        # Corner 1: intersection of treats and owl-fraction constraints
        # 3x + 5y = 1000
        # 3y = 2x  ->  x = 1.5y
        # Substitute: 3(1.5y) + 5y = 1000 -> 4.5y + 5y = 1000 -> 9.5y = 1000
        y1 = 1000.0 / 9.5
        x1 = 1.5 * y1

        # Check x >= 20 (it is, but we keep general)
        if x1 < 20:
            # If this happened, we'd need to adjust, but numerically it doesn't.
            pass

        L1 = 2 * x1 + 5 * y1

        # Corner 2: intersection of treats constraint with x = 20
        # 3*20 + 5y = 1000 -> 60 + 5y = 1000 -> 5y = 940 -> y = 188
        x2 = 20.0
        y2 = 188.0
        # Check owl-fraction: y <= (2/3)x -> 188 <= 13.333... (violated), so infeasible.
        # We won't use this point.

        # Corner 3: intersection of owl-fraction with x = 20
        # y = (2/3)*20 = 40/3
        x3 = 20.0
        y3 = (2.0 / 3.0) * 20.0
        # Check treats: 3x + 5y <= 1000
        if 3 * x3 + 5 * y3 <= 1000:
            L3 = 2 * x3 + 5 * y3
        else:
            L3 = float('-inf')

        # Corner 4: x = 20, y = 0
        x4 = 20.0
        y4 = 0.0
        L4 = 2 * x4 + 5 * y4

        # The feasible region is bounded by these constraints; the optimum is at one of these corners.
        best = max(L1, L3, L4)

        return float(best)

Strong PoT:
  pred          = 841.0
  correct       = True
  abs_error     = 0.0
  rel_error     = 0.0
  latency_sec   = 8.517687320709229
================================================================================
[FIXED] Example index = 11
================================================================================
Question:
  A village hosts a banquet and provides bike and car transportation for everyone. A bike can take 3 people while a car can take 5 people. Since cars are more expensive, at most 40% of the vehicles can be cars. If the village needs to transport at least 500 people, how many of each vehicle should be used to minimize the total number of bikes needed?

Ground truth objective: 80.0

Baseline PoT:
  parsed_answer = 78.94736842105263
  correct       = False
  abs_error     = 1.05263157894737

  Baseline PoT code snippet:
    def solve():
        # Let b = number of bikes, c = number of cars
        # Constraints:
        # 1) 3b + 5c >= 500  (capacity)
        # 2) c <= 0.4(b + c) => c <= (2/3)b => 3c <= 2b
        # 3) b, c >= 0
        # Objective: minimize b
    
        # From 3c <= 2b, we get b >= 1.5c.
        # To minimize b for a given c, set b = 1.5c.
        # Substitute into capacity: 3(1.5c) + 5c = 4.5c + 5c = 9.5c >= 500
        # => c >= 500 / 9.5
        c = 500.0 / 9.5
        b = 1.5 * c
    
        # The problem does not require integrality, so we return the minimal b as float.
        return b

Strong PoT:
  pred          = 80.0
  correct       = True
  abs_error     = 0.0
  rel_error     = 0.0
  latency_sec   = 9.8647358417511
================================================================================
[FIXED] Example index = 20
================================================================================
Question:
  A bakery bakes bagels and croissants. A batch of bagels can be made using 2 hours of oven time and 0.25 hours of pastry chef time. A batch of croissants is more complicated, so while they take 1 hour of oven time, they take 2 hours of pastry chef time. In a day, the bakery has at most 70 hours available for the oven and 32 pastry chef hours available. Using all the available capacity, what is the maximum profit the bakery can generate assuming the profit per batch is $20 and $40 respectively for a batch of bagels and a batch of croissants.

Ground truth objective: 1060.0

Baseline PoT:
  parsed_answer = 1072.0
  correct       = False
  abs_error     = 12.0

  Baseline PoT code snippet:
    def solve():
        # We solve the LP:
        # Maximize 20x + 40y
        # subject to:
        # 2x + 1y = 70   (oven fully used)
        # 0.25x + 2y = 32 (chef fully used)
        # x, y >= 0
    
        # Solve the 2x2 linear system:
        # 2x + y = 70
        # 0.25x + 2y = 32
    
        # From first: y = 70 - 2x
        # Substitute into second:
        # 0.25x + 2(70 - 2x) = 32
        # 0.25x + 140 - 4x = 32
        # -3.75x = -108
        # x = 108 / 3.75
        x = 108.0 / 3.75
        # y = 70 - 2x
        y = 70.0 - 2.0 * x
    
        # Objective: 20x + 40y
        profit = 20.0 * x + 40.0 * y
        return profit

Strong PoT:
  pred          = 1060.0
  correct       = True
  abs_error     = 0.0
  rel_error     = 0.0
  latency_sec   = 6.807861566543579

[regressed] Showing up to 3 examples...
================================================================================
[REGRESSED] Example index = 4
================================================================================
Question:
  A farmer wants to mix his animal feeds, Feed A and Feed B, in such a way that the mixture will contain a minimum of 30 units of protein and 50 units of fat. Feed A costs $100 per kilogram and contains 10 units of protein and 8 units of fat. Feed B costs $80 per kilogram and contains 7 units of protein and 15 units of fat. Determine the minimum cost of the mixture.

Ground truth objective: 327.6595744680851

Baseline PoT:
  parsed_answer = 327.6595744680851
  correct       = True
  abs_error     = 0.0

  Baseline PoT code snippet:
    def solve():
        # We solve the LP:
        # Minimize: 100x + 80y
        # Subject to:
        # 10x + 7y >= 30   (protein)
        #  8x +15y >= 50   (fat)
        # x, y >= 0
    
        # The optimal solution for a 2-variable LP with linear constraints
        # will occur at a corner (intersection of constraint lines or axes).
        # We check:
        # 1) Intersection of the two constraint lines
        # 2) Intersection with axes where feasible
    
        # 1) Solve the system:
        # 10x + 7y = 30
        #  8x +15y = 50
    
        # Solve via substitution / elimination:
        # Multiply first equation by 8, second by 10:
        # 80x + 56y = 240
        # 80x +150y = 500
        # Subtract first from second:
        # 94y = 260  =>  y = 260/94
        y = 260.0 / 94.0
        # Plug back into 10x + 7y = 30:
        # 10x = 30 - 7y
        x = (30.0 - 7.0 * y) / 10.0
    
        # Check feasibility (x, y >= 0 automatically holds from calculation here)
        # Compute cost at this intersection:
        cost_intersection = 100.0 * x + 80.0 * y
    
        # 2) Check axis intersections:
        # Case x = 0:
        # Protein: 7y >= 30 => y >= 30/7
        # Fat:    15y >= 50 => y >= 50/15
        y_x0 = max(30.0 / 7.0, 50.0 / 15.0)
        cost_x0 = 100.0 * 0.0 + 80.0 * y_x0
    
        # Case y = 0:
        # Protein: 10x >= 30 => x >= 30/10 = 3
        # Fat:     8x >= 50  => x >= 50/8
        x_y0 = max(30.0 / 10.0, 50.0 / 8.0)
        cost_y0 = 100.0 * x_y0 + 80.0 * 0.0
    
        # The optimal cost is the minimum of these feasible corner costs
        optimal_cost = min(cost_intersection, cost_x0, cost_y0)
        return optimal_cost

Strong PoT:
  pred          = 340.0
  correct       = False
  abs_error     = 12.340425531914889
  rel_error     = 0.0376623376622227
  latency_sec   = 6.786689519882202

[Multi-strategy performance on this example (10-example run)]
  - cot          | correct=False | parsed=3.0 | abs_err=365.0
  - cot_reflect  | correct=False | parsed=3.0 | abs_err=365.0
  - direct       | correct=False | parsed=350.0 | abs_err=18.0
  - pot          | correct=False | parsed=366.6666666666667 | abs_err=1.3333333333333144
  - pot_reflect  | correct=False | parsed=366.6666666666667 | abs_err=1.3333333333333144
  - tot          | correct=False | parsed=3.0 | abs_err=365.0
  - tot_reflect  | correct=False | parsed=3.0 | abs_err=365.0

================================================================================
[REGRESSED] Example index = 5
================================================================================
Question:
  A flooring company produces engineered hardwood and vinyl planks. Their sales forecasts show an expected demand of at least 20,000 square foot of hardwood and 10,000 square feet of vinyl planks each week. To satisfy a shipping contract, a total of at least 60,000 square feet of flooring much be shipped each week. Due to a labor shortage issue, no more than 50,000 square feet of hardwood and 30,000  square feet of vinyl  can be produced weekly. If a square foot of hardwood flooring yields a profit of $2.5 and a square foot of vinyl planks produces a $3 profit, how many of each type of flooring should be made weekly to maximize the company's profit?

Ground truth objective: 215000.0

Baseline PoT:
  parsed_answer = 215000.0
  correct       = True
  abs_error     = 0.0

  Baseline PoT code snippet:
    def solve():
        # We solve the LP:
        # Maximize: 2.5*x + 3*y
        # Subject to:
        #   20000 <= x <= 50000
        #   10000 <= y <= 30000
        #   x + y >= 60000
    
        # Since this is a 2-variable LP with simple bounds, we can enumerate
        # all corner points of the feasible region and pick the best.
    
        candidates = []
    
        # Bounds
        x_min, x_max = 20000.0, 50000.0
        y_min, y_max = 10000.0, 30000.0
    
        # Helper to check feasibility
        def feasible(x, y):
            return (x_min <= x <= x_max and
                    y_min <= y <= y_max and
                    x + y >= 60000.0)
    
        # List all potential corner points formed by intersecting constraints:
        # 1) (x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)
        pts = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max)
        ]
    
        # 2) Intersections with x + y = 60000:
        #    y = 60000 - x, with x at its bounds
        for x in [x_min, x_max]:
            y = 60000.0 - x
            pts.append((x, y))
        #    x = 60000 - y, with y at its bounds
        for y in [y_min, y_max]:
            x = 60000.0 - y
            pts.append((x, y))
    
        # Evaluate feasible points
        best_val = float("-inf")
        for x, y in pts:
            if feasible(x, y):
                val = 2.5 * x + 3.0 * y
                if val > best_val:
                    best_val = val
    
        return best_val

Strong PoT:
  pred          = nan
  correct       = False
  abs_error     = nan
  rel_error     = nan
  latency_sec   = 8.384089708328247

[Multi-strategy performance on this example (10-example run)]
  - cot          | correct=True | parsed=582.0 | abs_err=0.0
  - cot_reflect  | correct=True | parsed=582.0 | abs_err=0.0
  - direct       | correct=False | parsed=540.0 | abs_err=42.0
  - pot          | correct=True | parsed=582.0 | abs_err=0.0
  - pot_reflect  | correct=False | parsed=46.0 | abs_err=536.0
  - tot          | correct=True | parsed=582.0 | abs_err=0.0
  - tot_reflect  | correct=False | parsed=46.0 | abs_err=536.0

================================================================================
[REGRESSED] Example index = 7
================================================================================
Question:
  A chair produced by Elm Furniture yields a profit of $43, while every dresser yields a $52 profit. Each week, 17 gallons of stain and 11 lengths of oak wood are available. Each chair requires 1.4 gallons of stain and 2 lengths of oak wood, while each dresser requires 1.1 gallons of stain and 3 lengths of oak wood. Determine the maximum profit.

Ground truth objective: 236.5

Baseline PoT:
  parsed_answer = 236.5
  correct       = True
  abs_error     = 0.0

  Baseline PoT code snippet:
    def solve():
        # We solve the LP:
        # Maximize 43x + 52y
        # subject to:
        # 1.4x + 1.1y <= 17   (stain)
        # 2x   + 3y   <= 11   (wood)
        # x, y >= 0
    
        # The maximum of a linear program with two variables occurs at a corner point
        # of the feasible region. The corner points are:
        # 1) Intersection with axes:
        #    - (0,0)
        #    - Stain-only: 1.4x = 17 -> (17/1.4, 0)
        #    - Wood-only: 2x = 11   -> (11/2, 0)
        #    - Stain-only: 1.1y = 17 -> (0, 17/1.1)
        #    - Wood-only: 3y = 11   -> (0, 11/3)
        # 2) Intersection of the two constraints:
        #    1.4x + 1.1y = 17
        #    2x   + 3y   = 11
    
        # We only keep points that satisfy both constraints and x,y >= 0.
    
        # Helper to check feasibility
        def feasible(x, y):
            if x < 0 or y < 0:
                return False
            if 1.4 * x + 1.1 * y > 17 + 1e-9:
                return False
            if 2 * x + 3 * y > 11 + 1e-9:
                return False
            return True

        # Objective function
        def profit(x, y):
            return 43 * x + 52 * y

        candidates = []

        # (0,0)
        candidates.append((0.0, 0.0))

        # x-intercepts
        # From stain: 1.4x = 17
        x1 = 17.0 / 1.4
        if feasible(x1, 0.0):
            candidates.append((x1, 0.0))

        # From wood: 2x = 11
        x2 = 11.0 / 2.0
        if feasible(x2, 0.0):
            candidates.append((x2, 0.0))

        # y-intercepts
        # From stain: 1.1y = 17
        y1 = 17.0 / 1.1
        if feasible(0.0, y1):
            candidates.append((0.0, y1))

        # From wood: 3y = 11
        y2 = 11.0 / 3.0
        if feasible(0.0, y2):
            candidates.append((0.0, y2))

        # Intersection of:
        # 1.4x + 1.1y = 17
        # 2x   + 3y   = 11
        # Solve the system using substitution or elimination.
        # From second: x = (11 - 3y)/2
        # Substitute into first:
        # 1.4*(11 - 3y)/2 + 1.1y = 17
        # 0.7*(11 - 3y) + 1.1y = 17
        # 7.7 - 2.1y + 1.1y = 17
        # 7.7 - 1.0y = 17
        # -y = 9.3 -> y = -9.3 (infeasible since y >= 0)
        y_int = -9.3
        x_int = (11.0 - 3.0 * y_int) / 2.0
        # Check feasibility (it will fail due to negative y)
        if feasible(x_int, y_int):
            candidates.append((x_int, y_int))

        # Evaluate profits at all feasible candidates
        best = float('-inf')
        for x, y in candidates:
            val = profit(x, y)
            if val > best:
                best = val

        return float(best)

Strong PoT:
  pred          = 224.0
  correct       = False
  abs_error     = 12.5
  rel_error     = 0.052854122621341
  latency_sec   = 7.0119030475616455

[Multi-strategy performance on this example (10-example run)]
  - cot          | correct=True | parsed=342.0 | abs_err=0.0
  - cot_reflect  | correct=True | parsed=342.0 | abs_err=0.0
  - direct       | correct=False | parsed=320.0 | abs_err=22.0
  - pot          | correct=True | parsed=342.0 | abs_err=0.0
  - pot_reflect  | correct=False | parsed=346.3414634146342 | abs_err=4.341463414634177
  - tot          | correct=False | parsed=41.0 | abs_err=301.0
  - tot_reflect  | correct=True | parsed=342.0 | abs_err=0.0


[both_wrong] Showing up to 3 examples...
================================================================================
[BOTH_WRONG] Example index = 15
================================================================================
Question:
  A hospital can transport their patients either using a type II ambulance or hospital van. The hospital needs to transport 320 patients every day. A type II ambulance is mounted on a truck-style chassis and can move 20 patients every shift and costs the hospital (including gas and salary) $820. A hospital van can move 15 patients and costs the hospital $550 every shift. The hospital can have at most 60% of shifts be hospital vans due to union limitations of the type II ambulance drivers. How many of shift using each type of vehicle should be scheduled to minimize the total cost to the hospital?

Ground truth objective: 12860.0

Baseline PoT:
  parsed_answer = 12385.882352941177
  correct       = False
  abs_error     = 474.1176470588253

  Baseline PoT code snippet:
    def solve():
        # We need to solve the LP:
        # Minimize: 820*A + 550*V
        # Subject to:
        #   20A + 15V >= 320        (patient requirement)
        #   V <= 0.6(A + V)         (at most 60% of shifts are vans)
        #   A >= 0, V >= 0
        #
        # Convert the 60% constraint:
        #   V <= 0.6A + 0.6V
        #   0.4V <= 0.6A
        #   V <= 1.5A
        #
        # So constraints:
        #   20A + 15V >= 320
        #   V <= 1.5A
        #   A >= 0, V >= 0
        #
        # In 2D, the optimum of a linear program over a convex polyhedron
        # occurs at a vertex (intersection of constraints).
        #
        # Candidate vertices:
        # 1) Intersection of 20A + 15V = 320 and V = 1.5A
        # 2) Intersection of 20A + 15V = 320 and V = 0 (if feasible)
        # 3) Intersection of 20A + 15V = 320 and A = 0 (if feasible)
        #
        # Check each, enforce nonnegativity and V <= 1.5A, then compute cost.

        best_cost = None

        # 1) Intersection of 20A + 15V = 320 and V = 1.5A
        # Substitute V = 1.5A:
        #   20A + 15*(1.5A) = 320
        #   20A + 22.5A = 320
        #   42.5A = 320
        A1 = 320.0 / 42.5
        V1 = 1.5 * A1
        if A1 >= 0 and V1 >= 0 and V1 <= 1.5 * A1 and 20 * A1 + 15 * V1 >= 320 - 1e-9:
            cost1 = 820 * A1 + 550 * V1
            best_cost = cost1

        # 2) Intersection of 20A + 15V = 320 and V = 0
        #   20A = 320 -> A = 16
        A2 = 16.0
        V2 = 0.0
        if A2 >= 0 and V2 >= 0 and V2 <= 1.5 * A2 and 20 * A2 + 15 * V2 >= 320 - 1e-9:
            cost2 = 820 * A2 + 550 * V2
            if best_cost is None or cost2 < best_cost:
                best_cost = cost2

        # 3) Intersection of 20A + 15V = 320 and A = 0
        #   15V = 320 -> V = 320/15
        A3 = 0.0
        V3 = 320.0 / 15.0
        if A3 >= 0 and V3 >= 0 and V3 <= 1.5 * A3 and 20 * A3 + 15 * V3 >= 320 - 1e-9:
            cost3 = 820 * A3 + 550 * V3
            if best_cost is None or cost3 < best_cost:
                best_cost = cost3

        # Return the optimal objective value as float
        return float(best_cost)

Strong PoT:
  pred          = 12600.0
  correct       = False
  abs_error     = 260.0
  rel_error     = 0.0202177293934665
  latency_sec   = 7.915932178497314
================================================================================
[BOTH_WRONG] Example index = 18
================================================================================
Question:
  A repairman fixes washing machines and freezers. Each washing machine takes 30 minutes of inspection and 90 minutes of fixing time. Each freezer takes 20 minutes of inspection and 125 minutes of fixing time. The repairman has 5000 minutes available for inspection and 20000 minutes available in his schedule. If each washing machine repaired earns him $250 and each freezer repaired earns his $375, how many of each appliance should he fix to maximize his earnings?

Ground truth objective: 60000.0

Baseline PoT:
  parsed_answer = 51724.137931034486
  correct       = False
  abs_error     = 8275.862068965514

  Baseline PoT code snippet:
    def solve():
        # We solve the LP:
        # Maximize 250x + 375y
        # subject to:
        # 30x + 20y <= 5000   (inspection time)
        # 120x + 145y <= 20000 (total schedule time: 90+30=120 for washer, 125+20=145 for freezer)
        # x, y >= 0
    
        # The optimum of a 2-variable LP lies at a corner (feasible vertex).
        # Candidate corners come from intersections of constraints and axes.
    
        # 1) Intersection with axes:
        # a) x = 0:
        #    30*0 + 20y <= 5000  -> y <= 250
        #    120*0 + 145y <= 20000 -> y <= 20000/145
        y1 = min(250, 20000.0 / 145.0)
        x1 = 0.0
        val1 = 250.0 * x1 + 375.0 * y1
    
        # b) y = 0:
        #    30x <= 5000 -> x <= 5000/30
        #    120x <= 20000 -> x <= 20000/120
        x2 = min(5000.0 / 30.0, 20000.0 / 120.0)
        y2 = 0.0
        val2 = 250.0 * x2 + 375.0 * y2
    
        # 2) Intersection of the two constraints:
        # 30x + 20y = 5000
        # 120x + 145y = 20000
        # Solve the system:
        # From first: y = (5000 - 30x)/20
        # Substitute into second:
        # 120x + 145*(5000 - 30x)/20 = 20000
        # We'll compute directly.
        import math
    
        # Solve linear system using determinant method:
        # a1 x + b1 y = c1
        # a2 x + b2 y = c2
        a1, b1, c1 = 30.0, 20.0, 5000.0
        a2, b2, c2 = 120.0, 145.0, 20000.0
        det = a1 * b2 - a2 * b1
        if abs(det) > 1e-12:
            x3 = (c1 * b2 - c2 * b1) / det
            y3 = (a1 * c2 - a2 * c1) / det
            # Check feasibility (nonnegative and satisfy constraints within tolerance)
            if x3 >= -1e-9 and y3 >= -1e-9:
                # Clip tiny negatives to zero
                x3 = max(0.0, x3)
                y3 = max(0.0, y3)
                # Check constraints
                if (30.0 * x3 + 20.0 * y3 <= 5000.0 + 1e-6 and
                    120.0 * x3 + 145.0 * y3 <= 20000.0 + 1e-6):
                    val3 = 250.0 * x3 + 375.0 * y3
                else:
                    x3 = y3 = None
                    val3 = -math.inf
            else:
                x3 = y3 = None
                val3 = -math.inf
        else:
            x3 = y3 = None
            val3 = -math.inf
    
        # Take the best of the feasible vertices
        best_val = max(val1, val2, val3)
        return float(best_val)

Strong PoT:
  pred          = 51625.0
  correct       = False
  abs_error     = 8375.0
  rel_error     = 0.139583333333331
  latency_sec   = 5.637821674346924
================================================================================
[BOTH_WRONG] Example index = 26
================================================================================
Question:
  A fishery wants to transport their catch. They can either use local sled dogs or trucks. Local sled dogs and trucks can take different amount of fish per trip. Also, the cost per trip for sled dogs and truck is also differs. You should note that the budget has an upper limit and the number of sled dog trips must be less than the number of truck trips. Formulate an LP to maximize the number of fish that can be transported.
  Concrete Data:
  DogCapability = 100
  TruckCapability = 300
  DogCost = 50
  TruckCost = 100
  MaxBudget = 1000

Ground truth objective: -99999.0

Baseline PoT:
  parsed_answer = 3000.0
  correct       = False
  abs_error     = 102999.0

  Baseline PoT code snippet:
    def solve():
        # We formulate the LP:
        # Maximize: 100*x + 300*y
        # Subject to:
        #   50*x + 100*y <= 1000   (budget)
        #   x <= y                (sled dog trips less than or equal to truck trips)
        #   x >= 0, y >= 0
        #
        # This is a 2-variable LP, so we can solve it analytically by checking corner points.
        #
        # Constraints:
        # 1) 50x + 100y <= 1000  ->  x + 2y <= 20
        # 2) x <= y
        # 3) x >= 0, y >= 0
        #
        # Feasible region corners come from intersections of:
        #   - x = 0
        #   - y = 0
        #   - x = y
        #   - x + 2y = 20
        #
        # Check candidate points:
        # A: (0, 0)
        # B: intersection of x = 0 and x + 2y = 20 -> (0, 10)
        # C: intersection of x = y and x + 2y = 20 -> 3y = 20 -> (20/3, 20/3)
        #
        # Also check where x = y and y = 0 -> (0, 0) already included.
        #
        # Verify feasibility and compute objective at each candidate.

        candidates = [
            (0.0, 0.0),
            (0.0, 10.0),
            (20.0 / 3.0, 20.0 / 3.0),
        ]

        best_value = float('-inf')
        for x, y in candidates:
            # Check feasibility
            if x < 0 or y < 0:
                continue
            if 50 * x + 100 * y > 1000 + 1e-9:
                continue
            if x - y > 1e-9:  # x <= y
                continue

            value = 100 * x + 300 * y
            if value > best_value:
                best_value = value

        # best_value is the optimal objective value
        return float(best_value)

Strong PoT:
  pred          = 3000.0
  correct       = False
  abs_error     = 102999.0
  rel_error     = 1.0300003000029896
  latency_sec   = 6.769336462020874