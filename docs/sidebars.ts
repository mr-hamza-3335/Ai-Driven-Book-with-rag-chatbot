import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  bookSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'introduction',
      ],
    },
    {
      type: 'category',
      label: 'Foundations',
      collapsed: false,
      items: [
        'ros2-basics',
        'gazebo-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      collapsed: false,
      items: [
        'nvidia-isaac',
        'vla',
      ],
    },
    {
      type: 'category',
      label: 'Project',
      collapsed: false,
      items: [
        'capstone',
      ],
    },
  ],
};

export default sidebars;
