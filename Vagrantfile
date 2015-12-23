# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/trusty64"

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine.
  config.vm.network "forwarded_port", guest: 5000, host: 8000

  # Create a public network, which generally matched to bridged network.
  # Bridged networks make the machine appear as another physical device on
  # your network.
  # config.vm.network "public_network"

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  config.vm.synced_folder "./", "/vagrant",  :mount_options => ["dmode=755,fmode=755"]

  config.vm.provider "virtualbox" do |v|
    host = RbConfig::CONFIG['host_os']

    # Give VM 1/2 system memory & access to all cpu cores on the host
    if host =~ /darwin/
      cpus = `sysctl -n hw.ncpu`.to_i
      # sysctl returns Bytes and we need to convert to MB
      mem = `sysctl -n hw.memsize`.to_i / 1024 / 1024 / 2
    elsif host =~ /linux/
      cpus = `nproc`.to_i
      # meminfo shows KB and we need to convert to MB
      mem = `grep 'MemTotal' /proc/meminfo | sed -e 's/MemTotal://' -e 's/ kB//'`.to_i / 1024 / 4
    else
      cpus = 2
      mem = 1024
    end

    v.customize ["modifyvm", :id, "--memory", mem]
    v.customize ["modifyvm", :id, "--cpus", cpus]
  end

  config.vm.provision "fix-no-tty", type: "shell" do |s|
     s.privileged = false
     s.inline = "sudo sed -i '/tty/!s/mesg n/tty -s \\&\\& mesg n/' /root/.profile"
  end

  config.vm.provision "install-dependencies", type: "shell", inline: <<-SHELL
     sudo apt-get update
     sudo apt-get upgrade -y
     sudo apt-get dist-upgrade -y
     sudo apt-get install -y gcc git sox build-essential python2.7-dev python-setuptools python-pip libc6-dev-i386 postgresql libpq-dev libav-tools libavcodec-extra-53
     sudo pip install inflect numpy requests[security] simplejson jsonschema click functools32 tgt ipython psycopg2 pandas dotenv python-Levenshtein pydub
     sudo pip install git+git://github.com/Miserlou/SoundScrape.git@5dbd42b61fb550d60be3a238024ceaf0fd42d117
   SHELL
end
