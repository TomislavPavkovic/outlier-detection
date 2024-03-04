from pathlib import Path
import re

def obj2off(objpath, offpath):
    line = ""

    vset = []
    fset = []
    with open(objpath,'r') as f:
        lines = f.readlines()
    p = re.compile(r'/+')
    space = re.compile(r' +')

    for line in lines:
        tailMark = " "
        line = line+tailMark
        if line[0]!='v' and line[0]!='f' :
            continue

        parameters = space.split(line.strip())
        if parameters[0] == "v": 
                Point = []
                Point.append(eval( parameters[1]) )
                Point.append(eval( parameters[2]) )
                Point.append(eval( parameters[3]) )
                vset.append(Point)

        elif parameters[0] == "f":
                vIndexSets = []
                for i in range(1,len(parameters) ):
                    x = parameters[i]
                    ans = p.split(x)[0]
                    index = eval(ans)
                    index -= 1
                    vIndexSets.append(index)

                fset.append(vIndexSets)

    with open(offpath, 'w') as out:
        out = open(offpath, 'w')
        out.write("OFF\n")
        out.write(str(vset.__len__()) + " " + str(fset.__len__()) + " 0\n")
        for j in range(len(vset)):
            out.write(str(vset[j][0]) + " " + str(vset[j][1]) + " " + str(vset[j][2]) + "\n")

        for i in range(len(fset)):
            s = str(len( fset[i] ))
            for j in range( len( fset[i] ) ):
                s = s+ " "+ str(fset[i][j])
            s += "\n"
            out.write(s)

#path = Path("shapenet/data/04401088/1f565ab552dc89727e51366b0cf7747")
input_file  = 'shapenet/data/04401088/1f565ab552dc89727e51366b0cf7747/isosurf.obj'
output_file = 'shapenet/data/04401088/1f565ab552dc89727e51366b0cf7747/isosurf.off'
obj2off(input_file, output_file)