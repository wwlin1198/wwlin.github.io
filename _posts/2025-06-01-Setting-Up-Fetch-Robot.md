---
layout: post
title: "Setting Up the Fetch Robot 14.04 Indigo"
tags: [Robotics, Setup]
---

## Setting Up the Fetch Robot in Ubuntu 14.04 and ROS Indigo
Currently there is not much documentation on the Fetch Robot that still uses 14.04 as it has migrated over to 18.04 melodic. There is no longer support after melodic as the company has been acquired by Zebra. You will also find that the fetch robot website no longer is active and the documentation website is not hosted and they moved it to: [https://fetchrobotics.github.io/docs/](https://fetchrobotics.github.io/docs/). 

### Dependencies
- OpenRave 0.9 (Motion Planing with MoveIt! in 14.04 is very rough and glitches a lot) Guide [https://scaron.info/robotics/installing-openrave-on-ubuntu-14.04.html](https://scaron.info/robotics/installing-openrave-on-ubuntu-14.04.html)
- FCL 0.5.0 (Flexible Collision Library)
- Fetch Robot Packages [https://github.com/orgs/ZebraDevs/repositories?q=fetch](https://github.com/orgs/ZebraDevs/repositories?q=fetch)
- Multimaster Fkie - This is the best way to run scripts on the fetch as the computer on the fetch is not powerful. Keep in mind that GPUs and CPUs that support 14.04 stops at 1080 generation. 20 series cards and newer won't work with 14.04.  




### Problems You Might Encounter

#### PyTorch
I think it is possible to find an early version with pytorch that is able to be installed in 14.04 but without internet connection, it is difficult as the browsers in 14.04 is too out of date now. For now every function can be written in early version of Numpy.

#### OpenRave Error
If you run across the error:

<div class="code-block-container">
  <div class="code-block-header">
    <span>JavaScript</span>
  </div>
  <pre><code class="language-bash">If you run across the error

"/usr/bin/python:
symbol lookup error:
/home/yuchen/projects/devel/lib/openrave-0.9/or_urdf_plugin.so:
undefined symbol: _ZN8tinyxml211XMLDocumentCieBNS_10WhitespaceE"</code></pre>
</div>

The issue is that you have two tinyxml2 running. Uninstall one of them to see which one.

Sometimes you will see a mutex error. This is usually caused by sourcing the wrong catkin or ROS environment. You will need to resource everything and see if it works again. Make sure everything is ran in `/home/fetch/[NAME OF WORKSPACE]`. Don't rosrun or roslaunch the `/src` folder to maintain consistency and errors.

#### Muiltimaster Fkie Error
If you see warnings while launching multimaster fkie, it would cause the action
servers to not work 

#### Networking Error
To fix most of the errors, make sure that the hostnames are register in /etc/hosts

#### Other Errors
I found that ChatGPT does help solve a lot of package installing errors if you specify that you are using openrave 0.9, ubuntu 14.04, ROS Indigo. The other error causing it to not work would most likely be symlink errors. 
