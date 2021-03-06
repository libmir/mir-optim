project('mir-optim', 'd', version: '2.0.0', license: 'BSL-1.0')

description = 'Mir Optim - Nonlinear Optimizers'

subprojects = [
    'cblas-d',
    'lapack-d',
    'mir-algorithm',
    'mir-blas',
    'mir-core',
    'mir-lapack',
    'mir-random',
]

if target_machine.system() == 'linux'
    subprojects += 'mir-linux-kernel'
endif

has_cpp_headers = false

sources_list = [
    'mir/optim/boxcqp',
    'mir/optim/least_squares',
]

sources = []
foreach s : sources_list
    sources += 'source/' + s + '.d'
endforeach

add_project_arguments([
    '-preview=dip1008',
    '-lowmem',
], language: 'd')

required_deps = []

foreach p : subprojects 
    required_deps += dependency(p, fallback : [p, 'this_dep'])
endforeach

directories = ['source']

if has_cpp_headers
    directories += 'include'
endif

directories = include_directories(directories)

this_lib = library(meson.project_name(),
    sources,
    include_directories: directories,
    install: true,
    version: meson.project_version(),
    dependencies: required_deps,
)

this_dep = declare_dependency(
    link_with: [this_lib],
    include_directories: directories,
    dependencies: required_deps,
)

test_versions = ['mir_optim_test']

if has_cpp_headers
    install_subdir('include/',
        strip_directory :true,
        install_dir: 'include/',
    )
endif

install_subdir('source/',
    strip_directory : true,
    install_dir: 'include/d/' + meson.project_name(),
)

import('pkgconfig').generate(this_lib,
    description: description,
    subdirs: 'd/' + meson.project_name(),
)

mir_optim_dep = this_dep
mir_optim_lib = this_lib

test_subdirs = []
