**Write a Python script that models a Mobius strip using parametric equations and computes key geometric properties.**



**1. Requirements**

Define a MobiusStrip class that

**Accepts**:

Radius R (distance from the center to the strip)

Width w (strip width)

Resolution n (number of points in the mesh)

**Computes:**

A 3D mesh/grid of (x, y, z) points on the surface

Surface area (numerically, using integration or approximation)

Edge length (numerically along the boundary)


**2. Parametric Equation of Mobius Strip**

Use the parametric equations:

x(u,v)=(R+v⋅cos⁡(u2))⋅cos⁡(u)

y(u,v)=(R+v⋅cos⁡(u2))⋅sin⁡(u)

z(u,v)=v⋅sin⁡(u2)

**Where:**

u∈[0,2π]

v∈[−w/2,w/2]


**3. Deliverables**

Python script (clean, modular, commented).

3D plot or screenshot (if you include visualization).


**A short write-up explaining:**

How you structured the code

How you approximated surface area

Any challenges you faced


**This assignment tests:**

Parametric 3D modeling

Numerical integration / geometry

Visualization

Code clarity

