const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = (env, argv) => ({
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'static' },
      ],
    }),
    new HtmlWebpackPlugin({
      title: argv.mode === 'development' ? 'Development' : '',
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
