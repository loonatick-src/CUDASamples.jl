using CUDA

function topology_query()
  devices = CUDA.devices()
  device_count = length(devices)
  for device1 in devices
    for device2 in devices
      if device1 == device2
        continue
      end
      
    end
  end
end
