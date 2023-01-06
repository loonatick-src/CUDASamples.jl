using CUDA
using DataFrames

# TODO: There was an NVIDIA blog post on faster device query that is suitable for runtime
# Consider implementing that

# TODO: consider returning a `DataFrame` that cleans up the attribute names


"""
Loads all the device attribute enums along with their values into a dictionary.
Some attribute enums(codes) are not recognized by `attribute` and result in an error.
These attributes are returned as a set in the second argument.

Returns
-------
`device_attr_data :: Dict{CUdevice_attribute_enum, Int32}`

Key value pairs of device attribute and values their values for the input device.
"""
function device_query_table(device::CuDevice)
  # 
  all_device_attributes = instances(CUDA.CUdevice_attribute_enum)
  
  device_attr_data = Dict{eltype(all_device_attributes), Int32}()
  junk_attributes = Set{eltype(all_device_attributes)}()
  for  attr in all_device_attributes
    try
      v = attribute(device, attr)
      device_attr_data[attr] = v
    catch
      # apparently some of these enum variants (codes) are not recognized by `attribute`.
      # these are pushed into `junk_attributes`.
      push!(junk_attributes, attr)
    end
  end
  # device_attr_df = DataFrame(Attribute=all_device_attributes, Values = add_device_values)
  device_attr_data, junk_attributes
end
