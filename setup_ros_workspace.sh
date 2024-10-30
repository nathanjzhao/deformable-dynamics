# Fix broken packages and dependencies first
echo "Fixing any broken packages..."
sudo apt --fix-broken install -y
check_status "Fixing broken packages"

# Update and upgrade existing packages
sudo apt update && sudo apt upgrade -y
check_status "Updating system packages"

# Install dependencies in correct order
echo "Installing dependencies in order..."
sudo apt install -y \
    python3-catkin-pkg-modules \
    python3-rospkg-modules \
    python3-rosdep-modules \
    python3-rosdistro-modules \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential
check_status "Installing dependencies"
