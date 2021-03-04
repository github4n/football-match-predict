SIMPLEUI_HOME_INFO = False  # 是否显示simple ui版本信息
SIMPLEUI_CONFIG = {

    'system_keep': True,
    'menu_display': ['认证和授权', '比赛预测'],  # 开启排序和过滤功能
    'dynamic': True,  # 设置是否开启动态菜单, 默认为False
    'menus': [
        # 比赛预测(一级菜单)
        {
            'name': '比赛预测',
            'icon': 'el-icon-trophy',
            'models': [
                {
                    'name': '数据集',
                    'icon': 'fa fa-database',
                    'url': 'basic/matchdata/'
                },
                {
                    'name': '预测',
                    'url': '/predict-test/'
                },
            ],
        },
    ]
}
