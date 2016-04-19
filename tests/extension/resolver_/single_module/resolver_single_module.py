from __future__ import absolute_import
from __future__ import print_function
import sys
import os

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from veriloggen import *

def mkOrigLed():
    m = Module('blinkled')

    width = m.Parameter('WIDTH', 8)
    inc = m.Parameter('INC', 1)
    
    clk = m.Input('CLK')
    rst = m.Input('RST')
    led = m.OutputReg('LED', width)
    
    count = m.Reg('count', width + 10)

    m.Always(Posedge(clk))(
        If(rst)(
            count(0)
        ).Else(
            If(count == 1023)(
                count(0)
            ).Else(
                count(count + inc)
            )
        ))
    
    m.Always(Posedge(clk))(
        If(rst)(
            led( 0 )
        ).Else(
            If(count == 1023)(
                led(led + inc)
            )
        ))

    return m

def mkLed():
    led = mkOrigLed()
    return resolver.resolve_constant(led)

if __name__ == '__main__':
    led = mkLed()
    verilog = led.to_verilog()
    print(verilog)