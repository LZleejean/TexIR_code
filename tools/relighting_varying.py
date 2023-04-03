import os
import numpy as np
import cv2
import glob

from scipy.interpolate import interp1d



def change_color(frame_per_color, colors):
    new_colors = []
    new_colors.append(','.join(colors[0]))
    for i in range(1, len(colors)):
        x = np.array([0, 1])
        r = np.array([colors[i-1][0], colors[i][0]])
        f1 = interp1d(x, r)
        new_r = f1(np.linspace(0, 1, frame_per_color))

        g = np.array([colors[i-1][1], colors[i][1]])
        f2 = interp1d(x, g)
        new_g = f2(np.linspace(0, 1, frame_per_color))

        b = np.array([colors[i-1][2], colors[i][2]])
        f1 = interp1d(x, b)
        new_b = f1(np.linspace(0, 1, frame_per_color))

        for j in range(frame_per_color):
            new_colors.append('{},{},{}'.format(new_r[j], new_g[j], new_b[j]))
    
    return new_colors


# varying
lists=[6]
for index in lists:
    root_path = "/home/SecondDisk/Material/mitsuba-scenes/room_relighting/hdrhouse{}/varying".format(index)


    with open(os.path.join(root_path, 'cbox_pano_optix.xml'), 'r') as f:
        all_info = f.read()
    
    # # hdrhouse 1
    # colors = [
    #     list(map(str,[2.2, 12.38, 24.14])),
    #     list(map(str,[24.2, 2.38, 12.14])),
    #     list(map(str,[24.2, 2.38, 2.14])),
    #     list(map(str,[2.2, 24.38, 2.14])),
    #     list(map(str,[2.2, 24.38, 12.14])),
    #     list(map(str,[2.2, 12.38, 24.14]))
    # ]

    # hdrhouse 6
    colors = [
        list(map(str,[6.38, 12.14, 1.2])),
        list(map(str,[6.2, 1.38, 12.14])),
        list(map(str,[1.2, 12.38, 12.14])),
        list(map(str,[12.2, 12.38, 6.14])),
        list(map(str,[12.2, 1.38, 12.14])),
        list(map(str,[6.38, 12.14, 1.2]))
    ]


    frame_per_color = 5

    new_colors = change_color(frame_per_color, colors)

    for i in range(len(new_colors)):
        target_xml = os.path.join(root_path, 'cbox_pano_optix_{}.xml'.format(i))
        with open(target_xml, 'w') as f:
            f.write(all_info.replace('$$', new_colors[i]))
        
        cmd = "/home/lz/WorkSpace/code/OptixRender_lingli/OptixRenderer/build/bin/optixRenderer -f {} -o test_{}.png".format(target_xml,i)
        print(cmd)
        os.system(cmd)

        cmd = "/home/SecondDisk/Code/opensource/oidn/build/oidnDenoise --ldr {} --alb {} -o {}"\
            .format(os.path.join(root_path, 'test_{}_1.png'.format(i)),\
            os.path.join(root_path, 'test_{}baseColor_1.png'.format(i)),\
            os.path.join(root_path, 'test_denoise_{}.png'.format(i+1)) )
        os.system(cmd)


# moving
lists=[6]
for index in lists:
    root_path = "/home/SecondDisk/Material/mitsuba-scenes/room_relighting/hdrhouse{}/moving".format(index)


    with open(os.path.join(root_path, 'cbox_pano_optix.xml'), 'r') as f:
        all_info = f.read()
    # # hdrhouse 1
    # colors = [
    #     list(map(str,[-1.54852, -0.69413831, 6.56525])),
    #     list(map(str,[-1.54852, 0, 6.56525])),
    #     list(map(str,[-2.54852, 0, 5.56525])),
    #     list(map(str,[-2.54852, 0, 7.06525])),
    #     list(map(str,[-0.54852, 0, 7.06525])),
    #     list(map(str,[-1.54852, -0.69413831, 6.56525]))
    # ]

    # hdrhouse 6
    colors = [
        list(map(str,[-8.22709, -0.69413831, -3.45957])),
        list(map(str,[-8.22709, 0, -3.45957])),
        list(map(str,[-9.22709, 0, -3.95957])),
        list(map(str,[-9.22709, 0, -2.45957])),
        list(map(str,[-7.22709, 0, -2.45957])),
        list(map(str,[-8.22709, -0.69413831, -3.45957]))
    ]
    frame_per_color = 5

    new_colors = change_color(frame_per_color, colors)

    for i in range(len(new_colors)):
        target_xml = os.path.join(root_path, 'cbox_pano_optix_{}.xml'.format(i))
        rgb = new_colors[i].split(',')
        with open(target_xml, 'w') as f:
            f.write(all_info.replace('$x$', rgb[0]).replace('$y$', rgb[1]).replace('$z$', rgb[2]) )
        
        cmd = "/home/lz/WorkSpace/code/OptixRender_lingli/OptixRenderer/build/bin/optixRenderer -f {} -o test_{}.png".format(target_xml,i)
        print(cmd)
        os.system(cmd)

        cmd = "/home/SecondDisk/Code/opensource/oidn/build/oidnDenoise --ldr {} --alb {} -o {}"\
            .format(os.path.join(root_path, 'test_{}_1.png'.format(i)),\
            os.path.join(root_path, 'test_{}baseColor_1.png'.format(i)),\
            os.path.join(root_path, 'test_denoise_{}.png'.format(i+1)) )
        os.system(cmd)
