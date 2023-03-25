function decomposeIntoRectangles(env)::Vector{HyperRectangle}
    """
    Decompose the Rectangular environment into rectangular regions
    that avoid the rectangular obstacles
    -----------
    | I       |
    |    x    |
    |       T |
    -----------

    Trapezoidal Decomposition Algorithm (search on Wikipedia)
    -----------------------------------
    It scans the region with a vertical line.
    If it hits a corner of a convex polygon, it draws a vertical line
    at that point to decompose the regions.
    """

    xmin = env.workspace.lb[1]
    xmax = env.workspace.ub[1]
    ymin = env.workspace.lb[2]
    ymax = env.workspace.ub[2]

    rectangles::Vector{HyperRectangle} = []
    currY = ymax
    sortedObs = sort(env.obstacles, by = o -> o.ub[2], rev=true)

    # For each obstacle, we scan its top and its sides
    for (i, currO) in enumerate(sortedObs)

        xminObs = currO.lb[1]
        xmaxObs = currO.ub[1]
        yminObs = currO.lb[2]
        ymaxObs = currO.ub[2]

        ### 1. Top Region
        currXmin = xmin
        currXmax = xmax
        # Check if left & right handside hits any obstacle
        for otherO in setdiff(env.obstacles, [currO])
            if otherO.lb[2] < ymaxObs && ymaxObs < otherO.ub[2]
                # Left
                if currXmin < otherO.ub[1]
                    currXmin = otherO.ub[1]
                end
                # Right
                if otherO.lb[1] < currXmax
                    currXmax = otherO.lb[1]
                end
            end
        end
        lb = [currXmin, ymaxObs]
        ub = [currXmax, currY]
        push!(rectangles, HyperRectangle(lb, ub))

        ### 2. Sides
        # Check if left & right handside hits any obstacle
        currXmin = xmin
        currXmax = xmax
        for otherO in setdiff(env.obstacles, [currO])
            if otherO.lb[2] < yminObs && yminObs < otherO.ub[2]
                # Left
                if currXmin < otherO.ub[1]
                    currXmin = otherO.ub[1]
                end
                # Right
                if otherO.lb[1] < currXmax
                    currXmax = otherO.lb[1]
                end
            end
        end

        # Left
        lb = [currXmin, yminObs]
        ub = [xminObs, ymaxObs]
        push!(rectangles, HyperRectangle(lb, ub))
        # Right
        lb = [xmaxObs, yminObs]
        ub = [currXmax, ymaxObs]
        push!(rectangles, HyperRectangle(lb, ub))

        if i == length(sortedObs)
            currY = yminObs
        else
            ymaxNextObs = sortedObs[i+1].ub[2]
            if ymaxNextObs < yminObs && ymaxNextObs < ymaxObs
                currY = yminObs
            else
                currY = ymaxObs
            end
        end
    end

    # Lastly, add the bottom region that is left
    lb = [xmin, ymin]
    ub = [xmax, currY]
    push!(rectangles, HyperRectangle(lb, ub))

    return rectangles
end
