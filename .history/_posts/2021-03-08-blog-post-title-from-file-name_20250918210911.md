## Using the Turtlebot2 Robot in ROS Melodic

This is to document how I setup and ran the turtlebot2 with ROS Melodic (18.04). I was still using ROS1 but I will include ROS2 but it will be the same as the github that I have found. 

PLEASE HAVE FETCH CONNECTED THROUGH ETHERNET IF BASE IS NOT NEEDED AS IT LOSES CONNECTION IF IT IS ON WIFI IF NOT ACTIVELY USED DURING DEVELOPMENT

---

### Turtlebot2 Setup

#### Preliminaries
Before we start, make sure that all the workstations and robots have their own static IPs. If static IP is not implemented, please implement it using a separate router. Don’t use a school's static IP as it provides unstable Connection. If you are running multi-robot setup on ROS1, please use run the command: export ROBOT_NAME. 

#### Installing Packages

#### Running the Turtlebots
<div class="code-block-container">
  <div class="code-block-header">
    <span>Bash</span>
  </div>
  <pre><code class="language-bash">export ROS_MASTER_URI=http://&lt;robot_name_or_ip&gt;:11311</code></pre>
</div>


<div class="code-block-container">
  <div class="code-block-header">
    <span>Bash</span>
  </div>
  <pre><code class="language-bash">export ROS_MASTER_URI=http://robot_A:11311</code></pre>
</div>

```powershell
Write-Host "This is a powershell Code block";

# There are many other languages you can use, but the style has to be loaded first

ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```
