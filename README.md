# CloudGuard

Your monitoring sidekick &amp; security safe haven.


## Setup

Since most embedded computers are powered with ARM processors, we must compile OpenCV from source so we can use the libraries. While this could be [cross compiled](https://github.com/Itseez/opencv/tree/master/platforms) on a faster workstation, it's usually easier to compile everything on the native system itself. The setup steps below were written with intent for the [Odroid C1+](http://ameridroid.com/products/odroid-c1) running [Lubuntu](http://lubuntu.net/). Official precompiled versions of this image can be found in the [odroid repository](https://odroid.in/ubuntu_14.04lts/). There is also a [wiki page](http://odroid.com/dokuwiki/doku.php?id=en:odroid-c1) that expands on the [user manual](http://magazine.odroid.com/assets/manual/c1/pdf/odroid-c1-user-manual.pdf), and both can serve as helpful reference guides when getting everything up and running. Writing the image to an SD card is the most critical piece, and this can be done using the same steps as that for a [Rasberry Pi](https://www.raspberrypi.org/documentation/installation/installing-images/linux.md).


### Download the Code

Once the Odroid is running, install [git](https://git-scm.com/) and run the command below to download this repository. On Lubuntu, [git](https://git-scm.com/) is installed with `sudo apt-get -y install git`.

```
git clone https://github.com/jjones646/cloudguard
```

### Initalize Submodules

Now, go into the directory that was created above, and run the below command to pull in the source files for [OpenCV](https://github.com/Itseez/opencv) and [FFmpeg](https://github.com/FFmpeg/FFmpeg).

```
cd ./cloudguard && git submodule update --init
```

### Build & Install Libraries

Once the step above has finished downloading everything, you can then proceed to start compiling the libraries that will be needed for the CloudGuard to start.

```
./build/install-all.sh
```

### Start at Boot

If you'd like to have the CloudGuard program launch at startup of the device, an executable script will have to be added at `/etc/init.d/cloudguard`.

### Remotely Connecting

Since it's impractical to have a monitor always connected to the device, most find it easier to connect via `ssh` from another computer. When doing this, make you to specify the `-X` flag for forwarding back any windows from the program.

```
ssh -X odroid@<ip-address>
```

### That's It!

Well...kinda. If all of the above steps worked for you, then you no longer have to find that old keyboard and mouse in the bottom of your closet everytime you want to see the activity of the CloudGuard.

## Configure

The [`config.json`](./config.json) file can be modified for fine-tuning the camera to your exact environment. This is also where you can set limits on when and how often images will be delivered to an off-site server.

The easiest way to check what a connected camera is pointing at is to set the `window->enabled` value to `true`. This will open a window showing the camera's view at startup of the program. 
