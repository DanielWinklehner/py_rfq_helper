from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Display.SimpleGui import init_display

step_reader = STEPControl_Reader()
status = step_reader.ReadFile('./RFQ_noRMS_Spiral_Cut_90+angle_temp.stp')

if status == IFSelect_RetDone:  # check status
    failsonly = False
    step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
    step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)

    ok = step_reader.TransferRoot(1)
    _nbs = step_reader.NbShapes()
    aResShape = step_reader.Shape(1)
else:
    print("Error: can't read file.")
    sys.exit(0)

from OCC.Core.DataExchange import read_iges
    
display, start_display, add_menu, add_function_to_menu = init_display()
display.DisplayShape(aResShape, update=True)
start_display()
