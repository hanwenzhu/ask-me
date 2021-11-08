const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = (env, argv) => ({
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'static', to: 'static' },
      ],
    }),
    new HtmlWebpackPlugin({
      title: argv.mode === 'development' ? 'Ask Me (Development)' : 'Ask Me',
    }),
  ],
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  output: {
    clean: true,
  },
});
