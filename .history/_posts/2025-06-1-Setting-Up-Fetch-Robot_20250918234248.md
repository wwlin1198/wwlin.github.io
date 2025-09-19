---
layout: post
title: "Setting Up the Fetch Robot 14.04 Indigo"
---

## Setting Up the Fetch Robot in Ubuntu 14.04 and ROS Indigo


---

### Dependencies


### Known Issues
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