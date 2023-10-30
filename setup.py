from distutils.core import setup
setup(
  name = 'pallet_loading_env',         
  packages = ['pallet_loading_env'],   # Chose the same as "name"
  version = '0.1',      
  license='MIT',        
  description = 'An environment based on the OpenAI gym API tha can be used for the 3D-BPP',  
  author = 'Hendrik Jon Schmidt',                   
  author_email = 'hendrikjonschmidt@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/hndrjs/pallet_loading_env',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/hndrkjs/pallet_loading_env/archive/refs/tags/v_01.tar.gz',    # I explain this later on
  keywords = ['Weight Constraints', 'Stability-Check', '3D-BPP', 'Deep Reinforcement Learning'],   
  install_requires=[            # I get to this in a second
          'numpy',
          'pybullet',
          'gym'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Researchers',      # Define that your audience are developers
    'Topic :: Reinforcement Learning',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
