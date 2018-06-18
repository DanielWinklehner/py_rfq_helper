SetFactory('OpenCASCADE');

Point(1) = {0, 0, 0};
Point(2) = {1, 0.5, 0};
Point(3) = {2, 1.5, 0};
Point(4) = {3, 0.5, 0};
Spline(1) = {1, 2, 3, 4};
Wire(1) = {1};
Point(5) = {0, 0, -0.5};
Point(6) = {0, 0, 0.5};
Line(2) = {5, 6};

Extrude {Line{2};} Using Wire {1}