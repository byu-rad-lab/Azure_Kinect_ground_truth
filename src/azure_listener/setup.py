from setuptools import find_packages, setup

package_name = 'azure_listener'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='radlab',
    maintainer_email='radlab@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'test_azure_node = azure_listener.test_azure_node:main',
                             'filtered_pc = azure_listener.FilteredPointCloud:main',
                             'PC_extractor = azure_listener.PC_extractor:main'
        ],
    },
)
