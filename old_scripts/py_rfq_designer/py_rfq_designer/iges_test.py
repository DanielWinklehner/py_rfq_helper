#from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Extend.DataExchange import read_iges_file 
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Display.SimpleGui import init_display

TOLERANCE = 1e-6

class assert_isdone(object):
    '''
    raises an assertion error when IsDone() returns false, with the error
    specified in error_statement
    '''
    def __init__(self, to_check, error_statement):
        self.to_check = to_check
        self.error_statement = error_statement
        
    def __enter__(self, ):
        if self.to_check.IsDone():
            pass
        else:
            raise AssertionError(self.error_statement)
        
    def __exit__(self, assertion_type, value, traceback):
        pass

    
def intersect_shape_by_line(topods_shape, line, low_parameter=0.0, hi_parameter=float("+inf")):
    """
    finds the intersection of a shape and a line
    :param shape: any TopoDS_*
    :param line: gp_Lin
    :param low_parameter:
    :param hi_parameter:
    :return: a list with a number of tuples that corresponds to the number
    of intersections found
    the tuple contains ( gp_Pnt, TopoDS_Face, u,v,w ), respectively the
    intersection point, the intersecting face
    and the u,v,w parameters of the intersection point
    :raise:
    """
    from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
    shape_inter = IntCurvesFace_ShapeIntersector()
    shape_inter.Load(topods_shape, TOLERANCE)
    #shape_inter.PerformNearest(line, low_parameter, hi_parameter)
    
    with assert_isdone(shape_inter, "failed to computer shape / line intersection"):
        return (shape_inter.Pnt(1),
                shape_inter.Face(1),
                shape_inter.UParameter(1),
                shape_inter.VParameter(1),
                shape_inter.WParameter(1))

# display, start_display, add_menu, add_function_to_menu = init_display()

myshapes = read_iges_file("./RFQ_noRMS_Spiral_Cut_90+angle_temp.igs", return_as_shapes=False)
print(type(myshapes))
for shape in myshapes:
    print(type(shape))
    print()
    print(shape)
    
intersect_shape_by_line(myshapes[0], None)

#display.DisplayShape(myshapes, update=True)
#start_display()
